import torch
import torch.nn as nn
import torch.nn.functional as F

# --- HELPER BLOCKS ---

class ConvBlock(nn.Module):
    """Standard Conv + Batch Norm + SiLU"""
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    """
    Cross Stage Partial Block.
    Improves gradient flow and efficiency.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = ConvBlock(in_c, out_c // 2, 1, 1, 0)
        self.conv2 = ConvBlock(in_c, out_c // 2, 1, 1, 0)
        self.res_block = nn.Sequential(
            ConvBlock(out_c // 2, out_c // 2),
            ConvBlock(out_c // 2, out_c // 2)
        )
        self.conv_out = ConvBlock(out_c, out_c, 1, 1, 0)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.res_block(self.conv2(x))
        return self.conv_out(torch.cat([y1, y2], dim=1))

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network Node.
    Uses learnable weights for feature fusion.
    """
    def __init__(self, channels):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4
        
        self.conv_up = ConvBlock(channels, channels)
        self.conv_down = ConvBlock(channels, channels)

    def forward(self, p_up, p_down, p_m=None):
        # 1. Top-Down Path
        w1 = torch.relu(self.w1)
        w1 = w1 / (torch.sum(w1) + self.epsilon)
        
        p_up_resized = F.interpolate(p_up, size=p_down.shape[2:], mode='nearest')
        p_mid = self.conv_up(w1[0] * p_down + w1[1] * p_up_resized)
        
        if p_m is None: return p_mid
        
        # 2. Bottom-Up Path
        w2 = torch.relu(self.w2)
        w2 = w2 / (torch.sum(w2) + self.epsilon)
        
        p_mid_down = F.interpolate(p_mid, size=p_m.shape[2:], mode='nearest')
        p_out = self.conv_down(w2[0] * p_m + w2[1] * p_mid_down + w2[2] * p_down)
        return p_out

# --- MAIN MODEL ---

class FaceDetector(nn.Module):
    def __init__(self, advanced=False):
        super().__init__()
        self.advanced = advanced
        print(f"Model Mode: {'ADVANCED (CSP+BiFPN)' if advanced else 'BASELINE (Simple CNN+FPN)'}")

        # --- BACKBONE ---
        C_Block = CSPBlock if advanced else ConvBlock
        
        # Stem Layers
        self.stem = nn.Sequential(
            ConvBlock(3, 32, stride=2), 
            ConvBlock(32, 64, stride=2)
        )
        
        # Feature Layers (P3, P4, P5)
        self.layer_p3 = nn.Sequential(C_Block(64, 128), ConvBlock(128, 128, stride=2))
        self.layer_p4 = nn.Sequential(C_Block(128, 256), ConvBlock(256, 256, stride=2))
        self.layer_p5 = nn.Sequential(C_Block(256, 512), ConvBlock(512, 512, stride=2))

        # --- NECK ---
        if advanced:
            # Channel adjustment for BiFPN
            self.p5_to_bifpn = ConvBlock(512, 128, 1, 1, 0)
            self.p4_to_bifpn = ConvBlock(256, 128, 1, 1, 0)
            self.p3_to_bifpn = ConvBlock(128, 128, 1, 1, 0)
            
            self.bifpn = BiFPNBlock(128)
        else:
            # Standard FPN
            self.lat_p5 = ConvBlock(512, 128, 1, 1, 0)
            self.lat_p4 = ConvBlock(256, 128, 1, 1, 0)
            self.lat_p3 = ConvBlock(128, 128, 1, 1, 0)

        # --- HEAD (Decoupled) ---
        head_dim = 128
        self.heads = nn.ModuleList([
            nn.Sequential(
                ConvBlock(head_dim, head_dim),
                nn.Conv2d(head_dim, 5, 1) 
            ) for _ in range(3)
        ])

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        p3 = self.layer_p3(x) 
        p4 = self.layer_p4(p3)
        p5 = self.layer_p5(p4)
        
        features = []
        
        # Neck
        if self.advanced:
            p3_in = self.p3_to_bifpn(p3)
            p4_in = self.p4_to_bifpn(p4)
            p5_in = self.p5_to_bifpn(p5)
            
            # BiFPN Flow
            p4_w = self.bifpn(p5_in, p4_in) 
            p3_w = self.bifpn(p4_w, p3_in)
            
            features = [p3_w, p4_w, p5_in]
            
        else:
            p5_lat = self.lat_p5(p5)
            p4_lat = self.lat_p4(p4) + F.interpolate(p5_lat, scale_factor=2)
            p3_lat = self.lat_p3(p3) + F.interpolate(p4_lat, scale_factor=2)
            
            features = [p3_lat, p4_lat, p5_lat]

        # Head
        outputs = []
        for feat, head in zip(features, self.heads):
            out = head(feat) 
            outputs.append(out)
            
        return outputs

if __name__ == "__main__":
    print("Testing Model Structure...")
    dummy_img = torch.randn(2, 3, 640, 640)
    
    model_adv = FaceDetector(advanced=True)
    out_adv = model_adv(dummy_img)
    print(f"Output Shape: {out_adv[0].shape}")
    
    params = sum(p.numel() for p in model_adv.parameters())
    print(f"Total Parameters: {params:,}")