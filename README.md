# Heart Disease Prediction

This repository contains a project focused on predicting the likelihood of heart disease using various machine learning algorithms. The dataset includes multiple features that can be used to assess the risk of heart disease in individuals. This project demonstrates data preprocessing, feature selection, model building, and evaluation.

## 🗂 Project Overview

The goal of this project is to develop a predictive model that estimates the probability of a person having heart disease based on their health metrics. The project explores different machine learning techniques to provide a reliable and accurate prediction.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Modeling Approach](#-modeling-approach)
- [Prerequisites](#-prerequisites)
- [Usage](#-usage)
- [Future Work](#-future-work)
- [Contact](#-Contact)

## 📊 Dataset

The dataset used in this project contains the following features:
- **Age**: Age of the individual.
- **Sex**: Gender of the individual (1 = male, 0 = female).
- **Chest Pain Type (cp)**: Type of chest pain experienced by the individual.
- **Resting Blood Pressure (trestbps)**: Resting blood pressure in mm Hg.
- **Serum Cholesterol (chol)**: Serum cholesterol in mg/dl.
- **Fasting Blood Sugar (fbs)**: Whether fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
- **Resting ECG (restecg)**: Resting electrocardiographic results.
- **Max Heart Rate (thalach)**: Maximum heart rate achieved.
- **Exercise Induced Angina (exang)**: Exercise-induced angina (1 = yes, 0 = no).
- **ST Depression (oldpeak)**: Depression induced by exercise relative to rest.
- **Slope of ST Segment (slope)**: The slope of the peak exercise ST segment.
- **Number of Major Vessels (ca)**: Number of major vessels (0-3) colored by fluoroscopy.
- **Thalassemia (thal)**: Thalassemia condition (3 = normal, 6 = fixed defect, 7 = reversible defect).
- **Target**: Presence of heart disease (1 = yes, 0 = no).

## 🧠 Modeling Approach

The project employs various machine learning models to predict heart disease risk:
- **Data Preprocessing**: Handling missing values, feature scaling, and encoding categorical variables.
- **Exploratory Data Analysis (EDA)**: Analyzing the distribution of features and relationships with the target variable.
- **Model Building**: Implementing multiple classification models including Logistic Regression, Random Forest, and Support Vector Machine (SVM).
- **Model Evaluation**: Evaluating models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

### Model Details:
- **Baseline Model**: Logistic Regression for initial performance benchmark.
- **Advanced Models**: Random Forest, SVM, and Neural Networks.
- **Hyperparameter Tuning**: Utilizing Grid Search and Random Search for optimal hyperparameters.

## 🛠 Prerequisites

To run this notebook, you need to have the following libraries installed:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
  
## 🖥 Usage

1. **Data Preparation**: Follow the steps in the notebook to load and preprocess the dataset.
2. **Model Training**: Train different models by executing the respective cells in the notebook.
3. **Evaluation**: Evaluate the models on the test set and analyze the performance metrics.

## 🔍 Future Work

Some potential improvements for this project include:
- Implementing advanced feature engineering techniques to improve model performance.
- Exploring additional machine learning models such as XGBoost or ensemble techniques.
- Using deep learning models for potentially better accuracy and generalization.

## 📫 Contact

- **Email**: [kasodariya.r@northeastern.edu](mailto:kasodariya.r@northeastern.edu)
- **LinkedIn**: [Rohan Kasodariya](https://www.linkedin.com/in/rohankasodariya/)
- **GitHub**: [RohanKasodariya](https://github.com/RohanKasodariya)

Feel free to reach out if you have any questions or suggestions!

---

Thanks for checking out this project! 😊
