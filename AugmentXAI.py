# %% Imports
import pandas as pd
import numpy as np
from keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
import shap

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

# %% LIME Explainer
explainer_lime = LimeTabularExplainer(
    training_data=X_train_np,
    feature_names=X_train.columns.tolist(),
    class_names=['Healthy future', 'Risky Future'],
    mode='classification',
    discretize_continuous=True
)

def predict_fn(x):
    preds = model.predict(x)
    return np.concatenate([(1 - preds), preds], axis=1)  # shape: (n, 2)

# %% LIME Explain one instance
instance_index = 0
instance = X_test_np[instance_index]

exp = explainer_lime.explain_instance(
    data_row=instance,
    predict_fn=predict_fn,
    num_features=15
)

exp.show_in_notebook()
for feature, weight in exp.as_list():
    print(f"{feature}: {weight:.4f}")

# %% SHAP Explainer
# Initialize SHAP's JS visualization
shap.initjs()

# Predict function that outputs probabilities
# (Assumes you’ve defined this already)
# predict_fn = lambda x: model.predict(x)

# ⚙️ Build KernelExplainer (only once, and reuse it!)
kernel_explainer = shap.KernelExplainer(predict_fn, X_test.iloc[:50, :])

# 🔍 Pick any instance index (e.g., 299)
instance_idx = 299
X_instance = X_test.iloc[instance_idx:instance_idx+1]

# 🔁 Compute SHAP values for that instance
shap_values = kernel_explainer(X_instance)

# 🎯 Choose class index (for binary: 0 or 1)
class_idx = 1

# ✅ Plot SHAP force plot
shap.plots.force(
    base_value=shap_values.base_values[0, class_idx],
    shap_values=shap_values.values[0, :, class_idx],
    features=shap_values.data[0],
    feature_names=X_test.columns,
    matplotlib=True  # static plot for Jupyter/IDE
)

