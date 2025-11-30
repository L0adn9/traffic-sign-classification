# Traffic Sign Classification (GTSRB) – Deep Learning Project

## 📌 Overview

This project trains a Convolutional Neural Network (CNN) to classify traffic signs using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. (GTSRB)** dataset.

The objective is to build, train, evaluate, and visualize the performance of a deep learning model.

---

## 🎯 Objective

* Build a CNN model capable of classifying traffic sign images.
* Use TensorFlow/Keras (or PyTorch) to implement the model.
* Train the model using GTSRB or simulated mock data.
* Produce evaluation metrics including accuracy, loss curves, and confusion matrix.

---

## 📦 Requirements

To run this project, install the following packages:

```
tensorflow
numpy
matplotlib
seaborn
pandas
scikit-learn
opencv-python
```

You may include these in a `requirements.txt` file for easy installation.

## 🧰 Skills & Tools Used

* **Deep Learning** (CNNs)
* **TensorFlow/Keras**
* **Image Preprocessing & Data Augmentation**
* **Transfer Learning** (optional)
* **Matplotlib / Seaborn** for visualization

---

## 📂 Dataset

You can download the official **GTSRB dataset**, or provide mock/simulated data using the structure below.

https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

```
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
unzip gtsrb-german-traffic-sign.zip -d ./data/
```

### **Expected Directory Structure:**

You may download the official GTSRB dataset, or simulate it using the following structure:

```
./data/
├── 00000/
│   ├── 1.png
│   ├── 2.png
├── 00001/
│   ├── 1.png
│   ├── 2.png
```

Where each folder represents a traffic sign class.

---

## 📝 Notebook Contents

The provided Jupyter Notebook contains:

### ✔️ Data Loading & Preprocessing

* Reading images and labels
* Resizing
* Normalization
* Train/validation/test splits
* Data augmentation

### ✔️ Model Development

* Custom CNN architecture
* Compilation with appropriate loss and optimizer

### ✔️ Model Training

* Tracking training and validation accuracy
* Avoiding overfitting using regularization/augmentation

### ✔️ Evaluation

Includes:

* **Accuracy and loss plots**
* **Confusion matrix**
* Final test accuracy

---

## 📊 Visualizations

The notebook generates:

* Training vs validation loss curve
* Training vs validation accuracy curve
* Confusion matrix of predictions vs true labels

These help assess how well the model generalizes.

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/L0adn9/traffic-sign-classification.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Launch the notebook:

```
jupyter notebook cnn-project.ipynb
```

4. Ensure dataset is placed in the `data/` directory.

---

## 📁 Repository Structure

```
project/
├── cnn-project.ipynb       # Main notebook
├── README.md               # This file
├── requirements.txt        # Dependencies
└── data/                   # Dataset folder
```

---

## 🚀 Future Improvements

* Use full GTSRB dataset (50,000+ images) for better performance
* Implement model optimization (learning rate schedules, callbacks)
* Deploy the model as a web app (Flask/Streamlit)

---

## 🏁 Conclusion

This project demonstrates the full pipeline for traffic sign classification using deep learning, from dataset loading to evaluation with visualizations.

Feel free to contribute, optimize the model, or extend the project!
