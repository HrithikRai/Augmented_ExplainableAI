#  Explainable AI Dashboard for Medical Risk Prediction

This project is an interactive Streamlit dashboard that provides **transparent and interpretable predictions** for medical risk analysis using deep learning models.

##  Overview

We combine powerful machine learning with **explainability tools (SHAP & LIME)** to predict whether a patient is at risk (e.g., for cardiovascular disease) and justify **why** the model made that prediction.

Key Features:
-  Model predictions with confidence scores
-  SHAP and LIME feature importance visualizations
-  Custom **CIA Trust Score** combining prediction confidence and explainability metrics
-  Integrated ChatGPT-style AI doctor that answers questions about each prediction
-  Built using Keras, SHAP, LIME, and Streamlit

##  How It Works

- The model predicts health risk from patient data (based on the Framingham dataset).
- SHAP & LIME highlight **which features influenced the decision** and by how much.
- CIA score gives a single metric for **trustworthiness** of the explanation.
- An optional AI chatbot powered by OpenAI explains predictions in natural language.

##  Run It Locally

```bash
git clone https://github.com/HrithikRai/Augmented_ExplainableAI.git
cd xai-health-dashboard
pip install -r requirements.txt
streamlit run app.py
