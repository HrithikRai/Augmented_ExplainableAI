# %% Imports
import pandas as pd
import numpy as np
from keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
import shap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# %% Load model
model = load_model("model_name.keras")

# %% Load and preprocess data
from utils.helper_functions import Loader
loader = Loader(r"data/framingham.csv")
loader.load_data()
loader.preprocess()

X_train, X_test, y_train, y_test = loader.get_data_split()
X_train_np = X_train.values
X_test_np = X_test.values

# %% Define predict function
def predict_fn(x):
    preds = model.predict(x)
    return np.concatenate([(1 - preds), preds], axis=1)  # shape: (n, 2)

# %% LIME Explainer
explainer_lime = LimeTabularExplainer(
    training_data=X_train_np,
    feature_names=X_train.columns.tolist(),
    class_names=['Healthy future', 'Risky Future'],
    mode='classification',
    discretize_continuous=True
)

# %% LIME Explanation
instance_index = 0
instance = X_test_np[instance_index]

exp = explainer_lime.explain_instance(
    data_row=instance,
    predict_fn=predict_fn,
    num_features=15
)

print("\n🟢 LIME Explanation:")
for feature, weight in exp.as_list():
    print(f"{feature}: {weight:.4f}")

# %% SHAP Explanation
shap.initjs()
kernel_explainer = shap.KernelExplainer(predict_fn, X_test.iloc[:50, :])

instance_idx = 299  # Different instance from LIME for demo
X_instance = X_test.iloc[instance_idx:instance_idx+1]

shap_values = kernel_explainer(X_instance)

class_idx = 1  # For 'Risky Future' class
shap.plots.force(
    base_value=shap_values.base_values[0, class_idx],
    shap_values=shap_values.values[0, :, class_idx],
    features=shap_values.data[0],
    feature_names=X_test.columns,
    matplotlib=True
)

# %% Meta-Trust Score Calculation

# CONFIG
TOP_K = 10
α, β = 0.6, 0.4
γ1, γ2, γ3 = 0.4, 0.4, 0.2

# LIME processing
lime_expl = dict(exp.as_list())
lime_ranked = dict(sorted(lime_expl.items(), key=lambda x: abs(x[1]), reverse=True))
lime_top_k = set(list(lime_ranked.keys())[:TOP_K])

# SHAP processing
shap_vals_instance = shap_values.values[0, :, class_idx]
shap_dict = dict(zip(X_test.columns, shap_vals_instance))
shap_ranked = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))
shap_top_k = set(list(shap_ranked.keys())[:TOP_K])

# 1. Jaccard Similarity
jaccard_sim = len(shap_top_k & lime_top_k) / len(shap_top_k | lime_top_k)

# 2. Cosine Similarity of normalized importance weights
all_features = list(set(shap_dict.keys()).union(lime_expl.keys()))
shap_vec = np.array([shap_dict.get(f, 0) for f in all_features]).reshape(1, -1)
lime_vec = np.array([lime_expl.get(f, 0) for f in all_features]).reshape(1, -1)

scaler = MinMaxScaler()
shap_norm = scaler.fit_transform(shap_vec)
lime_norm = scaler.transform(lime_vec)
cos_sim = cosine_similarity(shap_norm, lime_norm)[0][0]

# 3. Fidelity Score (SHAP expected value ≈ LIME intercept)
shap_expected = shap_values.base_values[0, class_idx]
lime_intercept = exp.intercept[class_idx]
fidelity_score = 1 - abs(shap_expected - lime_intercept)
fidelity_score = np.clip(fidelity_score, 0, 1)

# 4. Final Meta-Trust Score
meta_trust_score = γ1 * jaccard_sim + γ2 * cos_sim + γ3 * fidelity_score

# %% Output Summary
print("\n📊 Meta-Trust Score Analysis")
print(f"🔍 Instance index (LIME): {instance_index}")
print(f"🔍 Instance index (SHAP): {instance_idx}")
print(f"\n✅ Jaccard Similarity (Top-{TOP_K}): {jaccard_sim:.2f}")
print(f"✅ Cosine Similarity (Normalized Weights): {cos_sim:.2f}")
print(f"✅ Fidelity Score (SHAP ≈ LIME): {fidelity_score:.2f}")
print(f"\n🧠 Final Meta-Trust Score: {meta_trust_score:.2f} (Range: 0=low trust, 1=high trust)")

# Optional interpretation
def interpret_score(score):
    if score >= 0.8:
        return "🟢 High Trust: Both SHAP and LIME strongly agree."
    elif score >= 0.5:
        return "🟡 Medium Trust: Partial agreement, review recommended."
    else:
        return "🔴 Low Trust: Explanations diverge significantly."

print(f"\n🧾 Interpretation: {interpret_score(meta_trust_score)}")
