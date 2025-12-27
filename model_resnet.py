import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --- HELPER BLOCKS ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BiFPNBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4
        self.conv_up = ConvBlock(channels, channels)
        self.conv_down = ConvBlock(channels, channels)

    def forward(self, p_up, p_down, p_m=None):
        w1 = torch.relu(self.w1)
        w1 = w1 / (torch.sum(w1) + self.epsilon)
        p_up_resized = F.interpolate(p_up, size=p_down.shape[2:], mode='nearest')
        p_mid = self.conv_up(w1[0] * p_down + w1[1] * p_up_resized)
        
        if p_m is None: return p_mid
        
        w2 = torch.relu(self.w2)
        w2 = w2 / (torch.sum(w2) + self.epsilon)
        p_mid_down = F.interpolate(p_mid, size=p_m.shape[2:], mode='nearest')
        p_out = self.conv_down(w2[0] * p_m + w2[1] * p_mid_down + w2[2] * p_down)
        return p_out

# --- RESNET BACKBONE MODEL ---

class FaceDetector(nn.Module):
    def __init__(self, advanced=True): 
        super().__init__()
        print("Model Mode: RESNET18 BACKBONE (BASELINE)")

        # 1. Backbone: Pretrained ResNet18
        # Using ImageNet weights for Transfer Learning
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Layer Extraction
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1 # 64 ch
        self.layer2 = resnet.layer2 # 128 ch (P3)
        self.layer3 = resnet.layer3 # 256 ch (P4)
        self.layer4 = resnet.layer4 # 512 ch (P5)
        
        # 2. Neck: Connect to BiFPN
        self.p5_to_bifpn = ConvBlock(512, 128, 1, 1, 0)
        self.p4_to_bifpn = ConvBlock(256, 128, 1, 1, 0)
        self.p3_to_bifpn = ConvBlock(128, 128, 1, 1, 0)
        
        self.bifpn = BiFPNBlock(128)

        # 3. Head
        head_dim = 128
        self.heads = nn.ModuleList([
            nn.Sequential(
                ConvBlock(head_dim, head_dim),
                nn.Conv2d(head_dim, 5, 1) 
            ) for _ in range(3)
        ])

    def forward(self, x):
        # Backbone
        x = self.layer0(x)
        x = self.layer1(x)
        p3 = self.layer2(x) # 128
        p4 = self.layer3(p3) # 256
        p5 = self.layer4(p4) # 512
        
        # Neck (BiFPN)
        p3_in = self.p3_to_bifpn(p3)
        p4_in = self.p4_to_bifpn(p4)
        p5_in = self.p5_to_bifpn(p5)
        
        p4_w = self.bifpn(p5_in, p4_in)
        p3_w = self.bifpn(p4_w, p3_in)
        
        features = [p3_w, p4_w, p5_in]
        
        # Head
        outputs = []
        for feat, head in zip(features, self.heads):
            outputs.append(head(feat))
            
        return outputs