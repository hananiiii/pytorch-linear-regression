# PyTorch Linear Regression

This project demonstrates a simple implementation of a **linear regression model** using **PyTorch**.

## Features
- Manual model creation with `nn.Module`
- Custom training and evaluation loops
- Data visualization with `matplotlib`
- Training vs. testing loss tracking
- save and load model 
--------------

# PyTorch Linear Regression - Exercise Project

This project implements a linear regression model from scratch using **PyTorch**, inspired by deep learning exercises.

## ✨ Features

- Synthetic dataset generation using the linear formula `y = weight * x + bias`
- CPU/GPU agnostic model training
- Model subclassing with `nn.Module` and `nn.Parameter`
- Training/Testing loop with loss tracking
- Visualizations using `matplotlib`
- Model saving and loading with `torch.save` and `torch.load`
- Verifying model predictions post-loading

## 📊 Dataset

The dataset is created using:
- Weight = 0.3
- Bias = 0.9
- 100 data points
- 80% training / 20% testing split

## 📈 Visualizations

- Training vs. testing dataset plots
- Prediction vs. actual comparison

## 🧠 Model

The model is built using:
```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))
    
    def forward(self, x):
        return self.weight * x + self.bias
