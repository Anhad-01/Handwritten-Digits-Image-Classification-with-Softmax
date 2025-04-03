# Handwritten-Digits-Image-Classification-with-Softmax

## Overview
This repository contains an implementation of **digit classification** using a **Softmax-based neural network** in PyTorch. The model is trained on the **MNIST dataset**, which consists of 28x28 grayscale images of handwritten digits (0-9). The goal is to classify each image into one of the 10 digit classes using a **single-layer Softmax classifier**.

## Dataset
- **MNIST**: A benchmark dataset containing **60,000 training images** and **10,000 test images** of handwritten digits.
- The dataset is available in `torchvision.datasets` and is automatically downloaded.
- Images are normalized and converted to tensors using `transforms.ToTensor()`.

## Model Architecture
This project uses a **Softmax regression model**:
- **Input Layer**: 784 neurons (28x28 flattened pixels)
- **Output Layer**: 10 neurons (one for each digit class)
- **Activation Function**: Softmax for multi-class classification
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Stochastic Gradient Descent (SGD)

## Training Details
- **Epochs**: 10
- **Batch Size**: 100
- **Learning Rate**: 0.01
- **Evaluation Metrics**: Accuracy and Loss

## Results
- **Maximum Accuracy Achieved**: **92.02%**
- **Minimum Loss Achieved**: **0.4093**
- **Test Accuracy** (after training): Reported in the output after execution

## Installation & Usage
### Prerequisites
Ensure you have Python 3.x installed along with the required dependencies:
```bash
pip install torch torchvision matplotlib numpy
```

### Running the Notebook
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/digit-classification-softmax.git
   cd digit-classification-softmax
   ```
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Digit_Classification_with_Softmax.ipynb
   ```

## Visualizations
The model's learned weights are visualized as **28x28 images**, showing how each neuron in the output layer responds to different digit classes.

## License
This project is open-source and available under the **MIT License**.

