# MNIST-Fashion

## Machine Learning Project

**Authors**: Ahmed Tlili, Cyrine Akrout, Ahmed Zguir  
**Institution**: EPFL  
**Date**: May 30, 2024

## Introduction

This repository contains the implementation of various machine learning methods aimed at classifying images from the Fashion-MNIST dataset into their respective categories using deep learning techniques. The models implemented include Multi-Layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs), and Transformer models, all developed using PyTorch.

The `main` script utilizes these classes to train a model on a specified dataset and predicts on given samples. Since the test data is not provided, the performance of the models is compared on the validation set, which constitutes 20% of the training data. Additionally, Principal Component Analysis (PCA) is implemented for dimensionality reduction to enhance the MLP model’s performance.

For a detailed walkthrough, check the [report.pdf](report.pdf) file.

## Data Preparation

To get started, you need to download the dataset features from the following link: [MNIST Fashion Dataset](https://drive.google.com/drive/folders/1Ns6g0Gajm1-ZXLHeJIifCXVbcmNGznQI?usp=sharing). Once downloaded, move it to this project folder.

## Performance Comparisons

| Model     | Accuracy       |
|-----------|----------------|
| MLP       | 84.7%          |
| MLP + PCA | 82.66%         |
| CNN       | 91.12%         |
| ViT       | 80%            |

- **MLP**: Multi-Layer Perceptrons without PCA.
- **MLP + PCA**: Multi-Layer Perceptrons with Principal Component Analysis.
- **CNN**: Convolutional Neural Network.
- **ViT**: Vision Transformer.

The CNN model achieved the highest accuracy of 91.12%, making it the best performing model among the three. This is likely due to CNNs being particularly effective at capturing spatial hierarchies in image data through convolutional layers.

## Usage

To run the models with optimized hyperparameters, use the following commands:

1. **MLP without PCA**
    ```bash
    python main.py --data dataset --method nn --nn_type mlp --lr 1e-3 --max_iters 100
    ```

2. **MLP with PCA**
    ```bash
    python main.py --data dataset --method nn --nn_type mlp --lr 1e-3 --max_iters 100 --use_pca --pca_d 100
    ```

3. **CNN**
    ```bash
    python main.py --data dataset --method nn --nn_type cnn --lr 1e-3 --max_iters 20
    ```

4. **Vision Transformer**
    ```bash
    python main.py --data dataset --method nn --nn_type transformer --lr 3e-4 --max_iters 10
    ```

## Conclusion

This project demonstrates the effective implementation and evaluation of different machine learning algorithms for image classification tasks. The results indicate that CNNs are particularly well-suited for the Fashion-MNIST dataset, achieving the highest accuracy.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
