# Customer Churn Prediction

## Project Overview
Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a telecom service.

This project uses Python, Scikit-learn, and Streamlit to build an interactive web application for predicting customer churn.

## Project Objectives
- Analyze customer data and identify churn patterns.
- Perform Exploratory Data Analysis (EDA).
- Build a Machine Learning classification model.
- Predict whether a customer is likely to churn.
- Develop an interactive web application using Streamlit.

## Technologies Used
- Python 3.13.5
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Model
**Algorithm:** Logistic Regression

**Preprocessing:**
- Missing value imputation
- Standard Scaling
- One-Hot Encoding

**Model Evaluation:**

| Metric | Result |
|---|---:|
| Accuracy | 73.81% |
| Churn Precision | 50% |
| Churn Recall | 78% |
| Churn F1-score | 0.61 |

## Project Structure

```text
Customer_Churn_Prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── app.py
├── train_model.py
├── data_analysis.py
├── churn_model.pkl
└── .gitignore
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/bala-studio/Customer_Churn_Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Customer_Churn_Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### 6. Add the Dataset

Place the `Telco-Customer-Churn.csv` dataset inside the `data` folder.

### 7. Train the Model (Optional)

```bash
python train_model.py
```

### 8. Run the Web Application

```bash
streamlit run app.py
```

## Application Features
- Customer information input
- Contract and billing information
- Machine Learning-based churn prediction
- Interactive Streamlit interface

## Future Improvements
- Improve model accuracy through hyperparameter tuning.
- Add churn probability prediction.
- Compare multiple Machine Learning algorithms.
- Improve dashboard visualization.

## Author
**Bala Studio**

GitHub: https://github.com/bala-studio

---

*This project was developed as a Machine Learning project for Customer Churn Prediction.*
