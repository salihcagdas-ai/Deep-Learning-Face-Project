import torch
import torch.nn as nn
import math

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Strides for P3, P4, P5
        self.strides = [8, 16, 32] 

    def box_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU).
        """
        # Convert center format to corner format (x1, y1, x2, y2)
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        # Intersection
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                     torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

        # Union
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-6)
        return iou

    def build_targets(self, p, targets, stride):
        """
        Anchor-Free Target Assignment:
        Map targets to specific grid cells.
        """
        batch_size, grid_h, grid_w, _ = p.shape 
        device = targets.device
        
        # Scale targets to grid size
        t_batch = targets[:, 0].long()
        t_box = targets[:, 2:] * torch.tensor([grid_w, grid_h, grid_w, grid_h], device=device)
        
        # Grid indices
        g_i = t_box[:, 0].long() # x index
        g_j = t_box[:, 1].long() # y index
        
        # Clamp to grid boundaries
        g_i = g_i.clamp(0, grid_w - 1)
        g_j = g_j.clamp(0, grid_h - 1)
        
        # Target mask
        obj_mask = torch.zeros((batch_size, grid_h, grid_w), device=device, dtype=torch.bool)
        
        # Filter valid indices
        valid_indices = t_batch < batch_size
        t_batch = t_batch[valid_indices]
        g_i = g_i[valid_indices]
        g_j = g_j[valid_indices]
        t_box = t_box[valid_indices]
        
        if len(t_batch) > 0:
            obj_mask[t_batch, g_j, g_i] = 1
        
        return obj_mask, t_box, t_batch, g_i, g_j

    def forward(self, preds, targets):
        device = preds[0].device
        box_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        
        # Iterate over scales (P3, P4, P5)
        for i, p in enumerate(preds):
            p = p.permute(0, 2, 3, 1) 
            
            # Match targets
            obj_mask, t_box, t_batch, g_i, g_j = self.build_targets(p, targets, self.strides[i])
            
            # --- 1. Classification Loss (Objectness) ---
            pred_conf = p[..., 0]
            target_conf = torch.zeros_like(pred_conf)
            target_conf[obj_mask] = 1.0
            
            # Focal Loss equivalent using pos_weight
            cls_loss += nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=device))(pred_conf, target_conf)

            # --- 2. Box Regression Loss ---
            if obj_mask.sum() > 0:
                pred_box = p[t_batch, g_j, g_i, 1:]
                
                # Decode predictions
                px = torch.sigmoid(pred_box[:, 0]) + g_i
                py = torch.sigmoid(pred_box[:, 1]) + g_j
                pw = torch.exp(pred_box[:, 2]) * 5 
                ph = torch.exp(pred_box[:, 3]) * 5
                
                pred_rects = torch.stack([px, py, pw, ph], dim=1)
                
                # IoU Loss
                iou = self.box_iou(pred_rects, t_box)
                box_loss += (1.0 - iou).mean()

        return (box_loss * 5.0) + cls_loss