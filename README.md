# 🧠 Digit Classifier From Scratch (NumPy + Tkinter)

This project is a complete handwritten digit recognition pipeline built **entirely from scratch** using **NumPy** for the neural network and **Tkinter** for a real-time drawing UI.

> 🛠 No TensorFlow. No PyTorch. Just raw math and NumPy.

<!-- --- -->

## 🚀 Features

- ✅ Neural Network built from scratch using only NumPy  
- ✅ Trained on MNIST dataset for digit recognition  
- ✅ Uses ReLU and Softmax activations  
- ✅ Supports L2 regularization and optimizer variants
- ✅ Dropout Layers for generalization
- ✅ Real-time prediction with Tkinter canvas UI  
- ✅ Live confidence bars for each digit prediction  
- ✅ Model weights saved/loaded using `.npz` files  

<!-- --- -->

## 🧠 Neural Network Architecture

```
Input: 784 (28x28 image flattened)
↓
Dense Layer (784 → 128) + ReLU
↓
Dense Layer (128 → 64) + ReLU
↓
Dense Layer (64 → 10) + Softmax
↓
Output: Probability scores for digits 0–9
```

<!-- --- -->

## 🎓 Training

- Dataset: MNIST (60k samples)
- Loss: Categorical Cross-Entropy  
- Accuracy after training: **~98%**  
- Training time: ~47 seconds (CPU)  

<!-- --- -->

## 🖼 Real-Time UI (Tkinter)

- Draw a digit (0–9) in the canvas.  
- Neural network predicts the digit in real time.  
- Confidence bars update live beside the canvas.  

<!-- --- -->

## 📁 Project Structure

```
.
├── app/
│   └── app.py
├── mnist-data/
│   ├── train-data
│   └── test-data
├── model/
│   ├── model.ipynb
│   ├── model.py
│   └── parameters.npz
└── utils/
    ├── Activation.py
    ├── Layer.py
    ├── Loss.py
    └── Optimizer.py
```

<!-- --- -->

<!-- ## ✨ Demo

![demo-gif](demo.gif)  -->