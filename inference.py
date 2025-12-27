import torch
import cv2
import numpy as np
from model_resnet import FaceDetector

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model_resnet.pth"
VIDEO_PATH = "test_video.mp4"
OUTPUT_PATH = "output_resnet.mp4"
CONF_THRESHOLD = 0.60
IOU_THRESHOLD = 0.4

def non_max_suppression(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, iou_threshold)
    if len(indices) > 0:
        return indices.flatten()
    return []

def process_frame(frame, model, device):
    h_orig, w_orig = frame.shape[:2]
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    boxes_list = []
    scores_list = []
    
    for i, pred in enumerate(outputs):
        _, _, grid_h, grid_w = pred.shape
        stride = 640 / grid_w 
        pred = pred.permute(0, 2, 3, 1).squeeze(0)
        conf = torch.sigmoid(pred[..., 0])
        mask = conf > CONF_THRESHOLD
        
        if mask.sum() == 0: continue
            
        grid_coords = torch.nonzero(mask)
        matched_preds = pred[mask]
        matched_conf = conf[mask]
        
        grid_y, grid_x = grid_coords[:, 0], grid_coords[:, 1]
        
        tx = torch.sigmoid(matched_preds[:, 1])
        ty = torch.sigmoid(matched_preds[:, 2])
        tw = torch.exp(matched_preds[:, 3]) * 5 
        th = torch.exp(matched_preds[:, 4]) * 5
        
        cx = (tx + grid_x) * stride
        cy = (ty + grid_y) * stride
        w = tw * stride
        h = th * stride
        
        x1 = cx - w/2
        y1 = cy - h/2
        
        scale_x = w_orig / 640
        scale_y = h_orig / 640
        
        x1 = x1 * scale_x
        y1 = y1 * scale_y
        w = w * scale_x
        h = h * scale_y
        
        for j in range(len(matched_conf)):
            boxes_list.append([int(x1[j]), int(y1[j]), int(w[j]), int(h[j])])
            scores_list.append(float(matched_conf[j]))

    if len(boxes_list) > 0:
        indices = non_max_suppression(boxes_list, scores_list, IOU_THRESHOLD)
        for idx in indices:
            x, y, w, h = boxes_list[idx]
            score = scores_list[idx]
            
            # Using Red for ResNet
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3) 
            cv2.putText(frame, f"ResNet: {score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    return frame

def main():
    print("Loading ResNet Model...")
    model = FaceDetector().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("ResNet model loaded.")
    except FileNotFoundError:
        print(f"ERROR: Model file '{MODEL_PATH}' not found.")
        return

    model.eval()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        processed_frame = process_frame(frame, model, DEVICE)
        out.write(processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()