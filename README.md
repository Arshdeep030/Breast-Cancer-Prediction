#  Breast Cancer Prediction using Machine Learning

This project aims to predict whether a breast tumor is **benign** or **malignant** based on medical imaging features. The solution includes training various machine learning models, evaluating their performance, selecting the best one, and deploying it using a **Gradio-based web interface**.


## Dataset Overview

The dataset used is the **Breast Cancer Wisconsin Diagnostic Dataset**, which includes 569 samples and 32 columns:

- **Target column**: `diagnosis`  
  - M = Malignant (cancerous)  
  - B = Benign (non-cancerous)

- **Features**:  
  30 real-valued attributes related to the characteristics of the cell nuclei present in the digitized image of a breast mass, such as:
  - `radius_mean`, `texture_mean`, `area_mean`, `smoothness_mean`, etc.
  
- **Dropped columns**:  
  - `id`: Unique identifier (not useful for prediction)  
  - `Unnamed: 32`: An empty column (dropped if present)


## Data Preprocessing

- **Label Encoding**:  
  The `diagnosis` column was encoded using `LabelEncoder`, mapping:
  - B → 0 (Benign)
  - M → 1 (Malignant)

- **Feature Scaling**:  
  Applied `StandardScaler` to normalize feature values to ensure all algorithms perform optimally.

- **Train-Test Split**:  
  Data was split into training and testing sets to evaluate model generalization.


## Machine Learning Models & Results

Seven ML models were trained and evaluated:

| Model               | Accuracy Score |
|--------------------|----------------|
| Logistic Regression | 97.37%         |
| Random Forest       | 96.49%         |
| Support Vector Classifier (**Selected**) | 98.25% |
| XGBoost             | 95.61%         |
| K-Nearest Neighbors | 94.74%         |
| Naive Bayes         | 96.49%         |
| MLP Classifier      | 97.37%         |

**Support Vector Classifier (SVC)** was selected for deployment due to its **highest accuracy**.

## Model Saving

- The best model (`SVC`) was saved using **Pickle** as `model.pkl`.
- The **StandardScaler** used for input transformation was also saved as `scaler.pkl` to ensure consistent input during inference.


## Deployment with Gradio

A user-friendly Gradio interface was created to interact with the model.

###  How it works:
- User enters 30 comma-separated values (numerical features).
- Input is scaled using the trained `StandardScaler`.
- Model predicts whether the tumor is **benign** or **malignant**.
- Output is shown as:
  - 🟢 Not Cancerous (Benign)
  - 🔴 Cancerous (Malignant)

