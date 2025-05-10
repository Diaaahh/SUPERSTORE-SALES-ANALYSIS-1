# Superstore Sales Prediction Using Machine Learning

## 📌 Project Description

This project aims to predict **Sales** using historical retail data from a Superstore dataset. It uses regression-based machine learning models to explore how different features like region, category, shipping mode, etc., influence sales. The project includes full preprocessing, modeling, evaluation, and visualization pipelines.

---

## 📁 Project Structure

machine-learning-project/
│
├── data/ # Raw and cleaned CSV files
├── notebooks/ # Jupyter notebooks with exploratory analysis and model training
├── src/ # Python scripts for preprocessing, modeling, evaluation
├── models/ # Saved ML models (Pickle/Joblib format)
├── outputs/ # Graphs, charts, and exported reports
├── requirements.txt # Python dependencies
├── README.md # Project overview
└── .gitignore # Files to ignore in Git


---

## 🔁 ML Pipeline

1. **Data Collection:** Superstore historical data
2. **Data Cleaning:** Handling nulls, duplicates, encoding categorical variables
3. **EDA:** Visual insights using seaborn and matplotlib
4. **Model Building:** Linear Regression, Decision Tree, Random Forest
5. **Model Evaluation:** Using MAE, MSE, RMSE, R² Score
6. **Model Saving:** Saved model files (optional)

---

## 📂 Dataset

- **Source:** Internal dataset (`SUPERSTORE-RAW DATASET.csv`)
- **Features:** Region, Category, Discount, Profit, Ship Mode, etc.
- **Target Variable:** `Sales`

---

## 🧠 Algorithms Used

| Algorithm           | MAE     | MSE       | RMSE     | R² Score |
|--------------------|---------|-----------|----------|----------|
| Linear Regression  | 78.52   | 23234.94  | 152.42   | 0.2941   |
| Decision Tree      | 29.84   | 5595.13   | 74.80    | 0.8663   |
| Random Forest      | 23.75   | 3762.78   | 61.35    | 0.9142   |

> ✅ **Random Forest** performed best with the lowest RMSE and highest R² Score.

---

## 📌 Tech Stack / Tools Used

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

---

## ▶️ How to Run the Project

1. Clone this repository  
```bash
git clone https://github.com/yourusername/superstore-ml-project.git

Navigate into the folder

bash
Copy
Edit
cd superstore-ml-project
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook or scripts

bash
Copy
Edit
jupyter notebook notebooks/superstore_analysis.ipynb
✍️ Author
Diya Nafren C

📌 Future Work
Hyperparameter tuning

XGBoost and ensemble stacking

Model deployment using Flask or Streamlit

yaml
Copy
Edit

---

Let me know if you want:
- The actual `requirements.txt` generated
- This file saved and exported for upload
- The above inserted directly into a GitHub repo for you
