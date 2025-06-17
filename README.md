#  Student Dropout Prediction Using Machine Learning
This project predicts whether a student will drop out of an online course based on demographic, academic, and behavioral data. It is based on the Open University Learning Analytics Dataset (OULAD), and uses classical ML models alongside explainability tools like SHAP.

---

##  Dataset
- **Source**: [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/#open-dataset)
- Includes:
  - Student demographics (age, gender, education level, region)
  - Academic data (number of previous attempts, credits studied)
  - Behavioral data (click logs, activity days, interactions with virtual learning environment)

---

##  Objective
To build a model that predicts student **dropout** (withdrawal from course) vs **completion** based on available features. This can help educational platforms **intervene early** and support at-risk students.

---

##  Models Used
- Logistic Regression (`class_weight=balanced`)
- Random Forest Classifier
- SHAP for model explainability

---

##  Workflow
1. **Data Preprocessing**
   - Merged multiple CSVs from OULAD
   - Created features like `total_clicks`, `active_days`, `avg_clicks`
   - One-hot encoded categorical variables
   - Filtered `final_result` to include `'Withdrawn'`, `'Fail'`, `'Pass'` and `'Distinction'`

2. **Model Training**
   - Logistic Regression: strong recall on dropout class (≈ 78%)
   - Random Forest: better AUC (≈ 0.87), slightly lower dropout recall

3. **Model Evaluation**
   - Metrics used: Accuracy, Precision, Recall, F1-score, AUC
   - Special focus on **recall for dropout class** (minimizing false negatives)

4. **Model Explainability**
   - Used SHAP values to visualize key drivers of dropout
   - Top features: `active_days`, `click_events`, `total_clicks`, `studied_credits`

   ---

   ## Key Results

| Metric            | Logistic Regression | Random Forest |
|------------------|---------------------|----------------|
| Accuracy          | 0.78                | 0.81           |
| Dropout Recall    | **0.78** ✅         | 0.57           |
| AUC Score         | 0.85                | **0.87** ✅     |

- SHAP plots revealed that **low engagement** (fewer clicks/days active) strongly correlates with higher dropout risk.
- Certain courses (`code_module_CCC`, etc.) showed higher dropout patterns.

---

