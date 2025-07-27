# Spanish Dialect Detector

This project detects regional Spanish dialects using audio samples. The audio data is preprocessed into 28×28 STFT (Short-Time Fourier Transform) spectrogram images, which are then used to train a Convolutional Neural Network (CNN) to classify the dialect.

## Project Overview

### Goal:
Build a machine learning model that classifies Spanish dialects based on speech audio samples using spectrogram-based image processing.

---

## How It Works

### 1. Data Preprocessing
- Audio files are read and transformed into spectrograms using the Short-Time Fourier Transform (STFT).
- The resulting spectrograms are downsampled or resized to 28x28 grayscale images.
- Each image is labeled according to its dialect region (e.g., Mexican, Argentine, Cuban, etc.).

### 2. Model Architecture
- A Convolutional Neural Network (CNN) is trained using the spectrogram images.
- The CNN learns to distinguish unique acoustic features corresponding to different dialects.
- The final layer outputs class probabilities for each dialect category.

### 3. Training and Evaluation
- The dataset is split into training and testing sets.
- Standard loss and accuracy metrics are used to evaluate performance.
- Training uses a typical optimizer (e.g., Adam or SGD) with cross-entropy loss.

---

## Tech Stack

- **Language**: Python
- **Libraries**:
  - `librosa` for audio processing
  - `matplotlib` and `PIL` for image generation
  - `NumPy`, `Pandas` for data manipulation
  - `PyTorch` (or `TensorFlow`) for CNN modeling
  - `scikit-learn` for splitting and metrics
