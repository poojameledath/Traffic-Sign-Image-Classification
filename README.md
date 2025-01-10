# Traffic Sign Recognition

A machine learning project that classifies traffic signs using image data. The project focuses on exploring and analyzing the Traffic Sign dataset, with experiments on additional datasets for further insights.

---

## Description

This project focuses on the development and implementation of machine learning techniques to classify images of traffic signs into their respective categories. It explores the end-to-end machine learning workflow, including data exploration, preparation, and modeling, while incorporating both traditional algorithms and modern deep learning methods.

---

## Table of Contents

- [Description](#description)
- [Dataset](#dataset)
- [Features ](#features)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Acknowledgment](#acknowledgement)
---



## Dataset

 The **Traffic Sign Dataset** contains images of different traffic signs, which are used for image classification tasks. It includes various categories of signs such as stop signs, yield signs, speed limits, and more, aimed at training models to recognize and classify these signs in real-world scenarios.
 
---
## Features

- Classification of traffic signs into multiple categories.
- Data preprocessing, including normalization and augmentation.
- Model training using a convolutional neural network (CNN).
- Comparison of performance with other models and datasets.
---
## Project Overview:
### Part 1: Data Exploration, Preprocessing, and Naïve Bayes Classification
In Part 1, we focused on exploring the dataset, preprocessing the data, performing feature selection, and applying the Naïve Bayes classifier. Below are the key findings:

#### Data Visualization and Exploration:
- We visualized the class distribution using a pie chart and examined the pixel intensity distribution, revealing that most pixels range from 0 to 50, indicating darker regions.
#### Image Enhancement:
- Applied Histogram Equalization and Gamma Correction to improve image contrast and brightness, enhancing the greyscale images for better feature extraction.
#### Data Preprocessing:
- Normalized pixel values by dividing by 255.
- Handled missing values (none found).
- Outliers were identified using DBSCAN, which detected 3,680 outliers.
#### Feature Selection:
- Used SelectKBest to retain the top 5, 10, and 20 features for each class, forming datasets with progressively fewer features for analysis.
#### Model Evaluation:
- Evaluated three Naïve Bayes classifiers (Gaussian, Multinomial, and Complement) using K-Fold Cross Validation. The models performed best with the top 5 features, with the Multinomial Naïve Bayes achieving an accuracy of 60.8%.

#### Confusion Matrix and ROC Curve:
- Generated confusion matrices and ROC curves for each model, showing that feature selection enhanced the model's performance, with the highest ROC AUC achieved using the top 5 features.

### Part 2: K-Means Clustering
In Part 2, we applied K-Means clustering to our dataset and evaluated its performance. Additionally, we explored various clustering algorithms (both hard and soft clustering), compared their effectiveness, and investigated methods for determining the optimal number of clusters.

#### Data Preprocessing:
- Principal Component Analysis (PCA) was used to reduce dimensionality, with 5 components chosen for the reduction.

#### K-Means Clustering:
- We applied K-Means to the complete dataset and the dataset with top 5 features, achieving accuracies of 35.72% and 53.08%, respectively.
- Visualization of clusters was performed using PCA, scatter plots, histograms, and violin plots.

#### Clustering Algorithms Explored:
- Hard Clustering: DBSCAN, BIRCH, and Agglomerative Clustering.
- Soft Clustering: Gaussian Mixture Model (GMM), Fuzzy C-Means, and Self-Organized Maps (SOM).
- PCA significantly improved clustering performance across all algorithms.
#### Optimal Number of Clusters:
- We evaluated the Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index to determine the best number of clusters, with the top 5 features dataset performing best.

#### Clustering vs. Bayesian Classification:
- K-Means clustering accuracy was compared with Bayesian models (Gaussian Naive Bayes and Multinomial Naive Bayes), showing that Naive Bayes models outperformed K-Means on the complete and top 5 features datasets.

### Part 3: Decision Tree Performance: Cross-Validation and Train-Test Classifier
In Part 3, we focused on training a Decision Tree classifier using different data splits and evaluated its performance. We also experimented with various decision tree parameters and compared it with other models.

#### Preprocessing:
- Image Enhancement: Applied techniques like histogram equalization, gamma correction, and normalization (dividing values by 255).
- Outlier Detection: Outliers were removed using DBScan.
- Oversampling: Used to balance class distribution in the training dataset.
#### Decision Tree Training:
- We trained the Decision Tree on different train-test splits (70:30, 60:40, 80:20) and evaluated accuracy.
- The model achieved a maximum accuracy of 94.85% with the 60:40 split.
- Cross-validation: Average accuracy of 92.57% from 10-fold cross-validation.

#### Evaluation Metrics:
- Metrics used: Accuracy, Precision, Recall, F1 Score, Sensitivity, Specificity, MAE, TP, TN, FP, FN rates.

#### Model Findings:
- Overfitting: A 100% accuracy on the training set indicated potential overfitting, with test accuracy dropping significantly (from ~94% to 71%).
- The model's performance varied with different splits, showing that larger test sets reduced overfitting.
#### Parameter Tuning:
- Experimented with parameters like max_depth, min_samples_leaf, and criterion to control tree complexity.
- The best model was identified using cross-validation with optimal parameters.
#### Comparison with Other Models:
- Random Forest and Extra Trees classifiers were tested alongside Decision Trees.
- Random Forest achieved the highest accuracy (99.36%) and cross-validation (98.31%), followed by Extra Trees (99.08%, 97.74%).
#### Conclusion:
- Decision Trees showed overfitting, but Random Forests outperformed all models.
- The decision tree's performance improved with different train-test splits, and further experiments with tree depth and other parameters helped reduce overfitting.

### Part 4: Neural Networks and Convolutional Neural Networks

In this section, we explore the performance of **Support Vector Machines (SVM)** as a linear classifier, applied to two datasets: one with **oversampling** and one without. The objective is to evaluate how well the model generalizes to new data and investigate whether the dataset is **linearly separable**.

---

#### Support Vector Machine (SVM) Performance Evaluation

##### Without Cross-Validation:
- We train the **SVM** on the training dataset without cross-validation and evaluate its performance on both the training and test sets.
- **Metrics:** Accuracy, Precision, Recall, F1 Score, Mean Absolute Error, etc.
- **Observations:** By comparing the performance on the training and test sets, we can identify potential issues like **overfitting** or **underfitting**.
- **Analysis:** High similarity between the training and test set performance suggests strong **generalization**. If there is a significant performance drop on the test set, it may indicate **overfitting** or that the data is not linearly separable.

##### With Cross-Validation:
To assess the stability and generalization of the model, we perform **10-fold cross-validation** on the dataset.
- **Metrics:** Accuracy from cross-validation, Precision, Recall, and F1 Scores.
- **Results:** Compare the **cross-validation performance** with the model's performance without cross-validation.

##### Comparison:
- We compare the results of **SVM without cross-validation** versus **SVM with cross-validation**.
- **Hypothesis:** If the SVM performs well on both training and test sets, this suggests the dataset is likely **linearly separable**. A large performance drop on the test set might indicate that the data is **not linearly separable** or that **overfitting** is happening.

---

#### Multilayer Perceptron (MLP) Evaluation

- **Experimentation:** Test different **hyperparameters** for the **MLP**, such as:
  1. **Activation Functions:** Tanh, ReLU, Logistic.
  2. **Hidden Layers and Nodes:** Experiment with varying the number of layers and nodes in each layer.
  3. **Learning Rate**, **Epochs**, **Momentum**, and **Validation Threshold**.
  
- **Hyperparameter Tuning:** 
  - Perform **Grid Search** and **Random Search** to identify the best parameter combinations.
  
- **Performance Evaluation:** 
  - After tuning the hyperparameters, evaluate the best configuration on both **training** and **test** datasets to determine the most effective model setup.


## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/poojameledath/Traffic-Sign-Image-Classification.git]

## Acknowledgments
Special thanks to my team members for their collaboration and contributions throughout this project. This work was a collective effort, and I truly appreciate everyone’s involvement and dedication.

