# Video Beauty Pipeline: An End-to-End Deep Learning Approach for Real-Time Facial Aesthetics Assessment in Video Streams

## Abstract
This project presents an end-to-end framework capable of assessing facial aesthetics in video streams in real-time. By utilizing the SCUT-FBP5500 dataset for facial beauty prediction, a lightweight deep neural network based on MobileNetV3-Small was fine-tuned for regression. The pipeline implements comprehensive steps spanning dataset preprocessing, model training, evaluation, export configurations for mobile deployment (ONNX/TFLite), and dynamic video inference. Faces that fall below configurable cosmetic scoring thresholds are dynamically filtered based on robust user-defined policies. This report documents the architecture, data management, and implementation specifics of the proposed system.

---

## 1. Introduction

### 1.1 Motivation
With the growing adoption of video conferencing and digital content creation, automated tools that can analyze, track, and conditionally filter environments or specific targets frame-by-frame are in high demand. Evaluating facial aesthetics quantitatively gives rise to dynamic filters where video feeds can selectively blur, augment, or hide participants based on algorithmic scores. Due to the real-time constraints of processing high-resolution video streams, large Transformer architectures or heavyweight CNNs are computationally prohibitive on edge devices.

### 1.2 Objective
The primary objective is to develop a highly efficient, real-time Machine Learning pipeline capable of:
1. Automatically extracting and aligning faces from image feeds.
2. Predicting facial beauty scores mapped to a predefined rating scale (1.0 to 5.0).
3. Analyzing full video files to generate dynamic overlays (annotations).
4. Providing a "cleaned" video output according to conditional policies (filtering out faces that do not meet the minimum score threshold).
5. Ensuring the architecture is lightweight enough to be eventually ported to mobile and edge platforms by utilizing TFLite exports.

---

## 2. Methodology & Architecture

### 2.1 Dataset Choice
The system was trained leveraging the **SCUT-FBP5500 dataset**, an established benchmark in Facial Beauty Prediction (FBP). 
To prevent the model from inadvertently learning correlations based on demographic differences rather than aesthetic features, the training set was deliberately constrained to specific geographic subsets. Only Caucasian Female (CF) and Caucasian Male (CM) images from the SCUT-FBP5500 dataset were utilized.

### 2.2 Data Preprocessing & Pipeline
Accurate aesthetic assessment demands strict geometrical alignment of the facial features before being processed by the neural network.
The data pipeline includes the following stages:
- **Face Extraction and Alignment:** `MediaPipe` is utilized to extract precise facial landmarks. Images are aligned via an affine transformation so the eyes and key features remain consistently scaled and horizontally balanced. The aligned outputs are cached to disk to massively accelerate subsequent experimentations.
- **Data Splitting Strategy:** To maintain the structural distribution of the target variable, a stratified train/validation/test split of **80/10/10** is employed. The target scalar scores were partitioned into 5 bins mathematically, assuring a balanced distribution of edge-cases (low/high-end scores) across all data folds.

### 2.3 Network Architecture
The core model is intentionally designed to be hardware-efficient:
- **Base Architecture:** `MobileNetV3-Small`, initialized with `ImageNet1k_V1` weights, acts as the primary feature extractor. MobileNetV3 leverages depthwise separable convolutions and Squeeze-and-Excitation networks to provide excellent feature mapping at a fraction of standard computational costs.
- **Regression Head:** The default 1000-class classifier was replaced with a custom feed-forward block projecting features down to a single scalar:
  - `Linear (in_features → 256)`
  - `Hardswish` Activation
  - `Dropout (p=0.2)`
  - `Linear (256 → 1)`
- **Loss Strategy:** As a pure regression task, the output acts as an unconstrained float during the training pass to prevent exploding/vanishing gradients caused by clipping operations. Output values are strictly clipped dynamically to `[1.0, 5.0]` limits strictly during inference.

---

## 3. Implementation Details

### 3.1 Training Pipeline
The training configurations strictly define reproducible deterministic states via early stopping mechanisms (`PATIENCE = 10` epochs) atop a base of `50` maximum iterations. Cross-validation logic loops frame evaluation per batch size (`BATCH_SIZE = 32`) normalized via global ImageNet statistics. Metrics like Mean Squared Error (MSE) and Pearson Correlation Coefficients supervise convergence.

### 3.2 Video Processing Module
A standalone video inferencing wrapper acts as the functional consumer of the trained model.
- **Batch Extractor:** Videos are converted to frames and selectively sampled by configurable parameters (e.g., `FRAME_SKIP` parameter set to skip standard 30 FPS rendering logic, saving GPU cycles).
- **Masking Mechanisms:** Frames featuring detected subjects are independently evaluated. If the regressive output for a face falls below a predetermined line (`SCORE_THRESHOLD = 3.0`), the system actions a fallback filter.
- **Filtering Policies:** Includes `any` (if a single person fails the test, mask the frame), `all` (filter applies only when all participants lack the score threshold), to `mean` representations.

### 3.3 Exporting & Optimization
Model portability bridges raw PyTorch experiments into functional APIs. A 4-step pipeline wraps the validation graph outputs directly to native ONNX states. Following standard topology validation through `onnxsim`, the system invokes `onnx2tf` generating compressed `TFLite` variants optimizing spatial tensors ready for mobile and iOS deployments seamlessly without custom binding rewrites.

---

## 4. Results & Artifact Generation
Running the pipeline generates distinct outputs proving the framework's operability:
1. **Network Weights:** Evaluated and checkpointed `best_model.pt`.
2. **Annotated Timelines:** Outputs a `_annotated.mp4` video with rendered bounding boxes, facial trajectories, and immediate model scores overlaid textually.
3. **Cleaned Stream:** Substantially modifies the original file producing a `_cleaned.mp4` compliant with the configured score policy.
4. **Distributions:** Plotting utilities report graphical representations summarizing score chronologies confirming aesthetic patterns through time series.

## 5. Conclusion
This project successfully synthesizes an end-to-end framework targeting Facial Beauty Prediction through optimized deep learning concepts. By integrating robust preprocessing pipelines reliant on MediaPipe and leveraging the MobileNetV3 backbone, the engine reaches parity between minimal computational delay and high prediction accuracy. The comprehensive support for PyTorch validation to TFLite deployment further anchors this project as a formidable baseline for next-generation, real-time cosmetic filters in commercial video applications.
