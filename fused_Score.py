# %% Imports
import numpy as np
import pandas as pd
from keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
import shap
from scipy.stats import entropy

# %% Load model and data
from utils.helper_functions import Loader
model = load_model("model_name.keras")

loader = Loader("data/framingham.csv")
loader.load_data()
loader.preprocess()

X_train, X_test, y_train, y_test = loader.get_data_split()
X_train_np, X_test_np = X_train.values, X_test.values

# %% Define prediction function
def predict_fn(x):
    preds = model.predict(x)
    return np.concatenate([(1 - preds), preds], axis=1)

# %% Choose instance to explain
instance_index = 12
instance_np = X_test_np[instance_index]
instance_df = X_test.iloc[[instance_index]]

# %% SHAP Explainer
shap.initjs()
explainer_shap = shap.Explainer(predict_fn, X_train_np[:100])
shap_vals = explainer_shap(instance_np.reshape(1, -1))
shap_vals_arr = shap_vals.values[0, :, 1]
shap_feat_importance = dict(zip(X_test.columns, shap_vals_arr))

# %% LIME Explainer
explainer_lime = LimeTabularExplainer(
    training_data=X_train_np,
    feature_names=X_train.columns.tolist(),
    class_names=['Healthy', 'Risky'],
    mode='classification'
)
lime_exp = explainer_lime.explain_instance(instance_np, predict_fn, num_features=15)
lime_feat_importance = dict(lime_exp.as_list())

# %% Global SHAP (optional summary)
#print("\n🌐 Global SHAP Importance:")
#shap.summary_plot(shap.Explainer(predict_fn, X_train_np)(X_test_np[:100]), feature_names=X_test.columns)

# %% CIA Score Calculation

# Confidence Score
pred_prob = model.predict(instance_np.reshape(1, -1))[0][0]
confidence = max(pred_prob, 1 - pred_prob)

# Interpretability Score (how top-heavy the explanation is)
def interpretability_score(imp_dict):
    vals = np.abs(list(imp_dict.values()))
    probs = vals / np.sum(vals)
    return 1 - entropy(probs) / np.log(len(probs))

interp_shap = interpretability_score(shap_feat_importance)
interp_lime = interpretability_score(lime_feat_importance)
interp_avg = (interp_shap + interp_lime) / 2

# Agreement Score: Top-k Jaccard or Kendall Tau
k = 10
top_shap = list(dict(sorted(shap_feat_importance.items(), key=lambda x: abs(x[1]), reverse=True)).keys())[:k]
top_lime = list(dict(sorted(lime_feat_importance.items(), key=lambda x: abs(x[1]), reverse=True)).keys())[:k]
jaccard = len(set(top_shap) & set(top_lime)) / len(set(top_shap) | set(top_lime))

# CIA Score
α, β, γ = 0.4, 0.3, 0.3
cia_score = α * confidence + β * interp_avg + γ * jaccard

# %% Output Summary
print("\n Instance:", instance_index)
print(f" Model Prediction: {pred_prob:.2f} → {'Risky' if pred_prob > 0.5 else 'Healthy'}")
print(f" Confidence Score: {confidence:.2f}")
print(f" Interpretability Score (avg of SHAP+LIME): {interp_avg:.2f}")
print(f" Explanation Agreement (Jaccard): {jaccard:.2f}")
print(f"\n Final CIA Score: {cia_score:.2f} (0=untrustworthy, 1=very trustworthy)")

# Feature breakdown
print("\n Top Features (SHAP):")
for f in top_shap[:5]:
    print(f"{f}: {shap_feat_importance[f]:.4f}")

print("\n Top Features (LIME):")
for f in top_lime[:5]:
    print(f"{f}: {lime_feat_importance[f]:.4f}")
