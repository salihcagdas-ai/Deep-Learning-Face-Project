# Real-Time Face Detection with CSP-BiFPN Architecture

This repository contains a PyTorch implementation of a custom real-time face detection model. The project compares a custom-designed architecture trained from scratch against a pre-trained ResNet18 baseline.

## 🧠 Model Architecture (Proposed Method)

The model is designed with a focus on real-time inference speed and small object detection capabilities. It features a custom **CSP-BiFPN** structure:

* **Backbone:** **CSP-Darknet** (Cross-Stage Partial Network) to optimize gradient flow and reduce computational cost.
* **Neck:** **BiFPN** (Bidirectional Feature Pyramid Network) implemented from scratch for weighted multi-scale feature fusion.
* **Head:** **Decoupled & Anchor-Free** detection head, separating classification and regression tasks.
* **Activation:** **SiLU** (Sigmoid Linear Unit) is used throughout the network.

## 📊 Performance Benchmark

We compared our custom model (trained from scratch) with a ResNet18 baseline (Transfer Learning from ImageNet).

| Model | Initialization | F1-Score | Inference Speed | Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet18 (Baseline)** | Pre-trained | **0.52** | Standard | Fast convergence, prone to overfitting on small data. |
| **CSP-BiFPN (Ours)** | **Scratch** | 0.42 | **Real-Time** | Stable generalization, no overfitting observed. |

*Note: While the baseline achieved a higher F1 score due to pre-training, the custom CSP-BiFPN model demonstrated better generalization stability during training (parallel Train/Val loss descent).*

## 📂 Project Structure

* `train.py`: Main training script with Early Stopping and Cosine Annealing LR.
* `model.py`: Implementation of the custom CSP-BiFPN architecture.
* `model_resnet.py`: Implementation of the ResNet18 baseline with FPN.
* `dataset.py`: Custom PyTorch Dataset class for loading images and YOLO-format labels.
* `loss.py`: Implementation of Focal Loss (Classification) and CIoU Loss (Regression).
* `inference.py`: Script for running inference on video files using the trained model.

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/REPO_NAME.git](https://github.com/YOUR_USERNAME/REPO_NAME.git)
    cd REPO_NAME
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Inference on Video:**
    Place your test video in the project directory as `test_video.mp4` and run:
    ```bash
    python inference.py
    ```
    *The output will be saved as `output_result.mp4`.*

4.  **Train the Model:**
    To train the model from scratch on your own dataset:
    ```bash
    python train.py
    ```

## 📈 Results

Training logs and charts (Loss and F1-Score progression) can be found in the `results/` directory.

---
*This project was developed for the Deep Learning course to demonstrate custom architecture design and optimization techniques.*