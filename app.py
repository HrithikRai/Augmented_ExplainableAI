import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
import shap
from scipy.stats import entropy
from openai import OpenAI
import matplotlib.pyplot as plt
from utils.helper_functions import Loader

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load model and data
model = load_model("model_name.keras")
loader = Loader("data/framingham.csv")
loader.load_data()
loader.preprocess()

X_train, X_test, y_train, y_test = loader.get_data_split()
X_train_np, X_test_np = X_train.values, X_test.values

# Prediction function
def predict_fn(x):
    preds = model.predict(x)
    return np.concatenate([(1 - preds), preds], axis=1)

# CIA score functions
def interpretability_score(imp_dict):
    vals = np.abs(list(imp_dict.values()))
    probs = vals / np.sum(vals)
    return 1 - entropy(probs) / np.log(len(probs))

def calculate_cia(instance_np, shap_dict, lime_dict, model):
    pred_prob = model.predict(instance_np.reshape(1, -1))[0][0]
    confidence = max(pred_prob, 1 - pred_prob)
    interp_shap = interpretability_score(shap_dict)
    interp_lime = interpretability_score(lime_dict)
    interp_avg = (interp_shap + interp_lime) / 2
    k = 10
    top_shap = list(dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)).keys())[:k]
    top_lime = list(dict(sorted(lime_dict.items(), key=lambda x: abs(x[1]), reverse=True)).keys())[:k]
    jaccard = len(set(top_shap) & set(top_lime)) / len(set(top_shap) | set(top_lime))
    α, β, γ = 0.4, 0.3, 0.3
    cia_score = α * confidence + β * interp_avg + γ * jaccard
    return pred_prob, confidence, interp_avg, jaccard, cia_score, top_shap, top_lime

# Sidebar
st.sidebar.title("🔍 Explainable AI Interface")
instance_index = st.sidebar.slider("Select Instance Index", 0, len(X_test) - 1, 12)

# Instance Selection
instance_np = X_test_np[instance_index]
instance_df = X_test.iloc[[instance_index]]

# SHAP
shap.initjs()
explainer_shap = shap.Explainer(predict_fn, X_train_np[:100])
shap_vals = explainer_shap(instance_np.reshape(1, -1))
shap_vals_arr = shap_vals.values[0, :, 1]
shap_feat_importance = dict(zip(X_test.columns, shap_vals_arr))

# LIME
explainer_lime = LimeTabularExplainer(
    training_data=X_train_np,
    feature_names=X_train.columns.tolist(),
    class_names=['Healthy', 'Risky'],
    mode='classification'
)
lime_exp = explainer_lime.explain_instance(instance_np, predict_fn, num_features=15)
lime_feat_importance = dict(lime_exp.as_list())

# CIA Score
pred_prob, confidence, interp_avg, jaccard, cia_score, top_shap, top_lime = calculate_cia(
    instance_np, shap_feat_importance, lime_feat_importance, model
)

# Title and Prediction
st.title("🧠 Explainable AI Dashboard")
st.markdown(f"""
- 🧬 **Instance Index**: `{instance_index}`
- 📊 **Model Prediction**: `{pred_prob:.2f}` → {'Risky' if pred_prob > 0.5 else 'Healthy'}
- ✅ **Confidence**: `{confidence:.2f}`
- 🔍 **Interpretability (Avg of SHAP & LIME)**: `{interp_avg:.2f}`
- 🤝 **SHAP-LIME Agreement**: `{jaccard:.2f}`
- 🧪 **CIA Trust Score**: `{cia_score:.2f}` _(0=Untrustworthy, 1=Very Trustworthy)_
""")

# Visualizations
st.subheader("📈 SHAP Feature Importances")
shap_df = pd.DataFrame(shap_feat_importance.items(), columns=["Feature", "Importance"])
shap_df = shap_df.reindex(shap_df.Importance.abs().sort_values(ascending=False).index)
st.bar_chart(shap_df.set_index("Feature"))

st.subheader("📈 LIME Feature Importances")
lime_df = pd.DataFrame(lime_feat_importance.items(), columns=["Feature", "Importance"])
lime_df = lime_df.reindex(lime_df.Importance.abs().sort_values(ascending=False).index)
st.bar_chart(lime_df.set_index("Feature"))

# Chatbot (Optional OpenAI/Custom Agent)
st.subheader("💬 AI Explanation Chat")
user_prompt = st.text_input("Ask something about this prediction...")
if user_prompt:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're an AI doctor and model explainer."},
            {"role": "user", "content": f"Explain this instance: {instance_df.to_dict()}"},
            {"role": "user", "content": user_prompt}
        ]
    )
    st.markdown("**AI Response:**")
    st.write(response.choices[0].message.content)
