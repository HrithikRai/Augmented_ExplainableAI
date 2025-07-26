import shap
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Load a sample dataset (for demonstration; replace with your own)
from utils.helper_functions import Loader
loader = Loader(r"data/framingham.csv")
loader.load_data()
loader.preprocess()

X_train, X_test, y_train, y_test = loader.get_data_split()
# 🚀 Load your pretrained DNN model
model = tf.keras.models.load_model("model_name.keras")

# 📊 Use KernelExplainer or DeepExplainer (based on data type)

# If your model is a DNN for tabular data:
explainer = shap.DeepExplainer(model, X_train[:100])  # use small background for speed

# 🔍 Select an instance to explain
instance_index = 0
instance = X_test.iloc[instance_index:instance_index+1]

# ✅ Compute SHAP values
shap_values = explainer.shap_values(instance)

# 📈 Plot force plot (for binary classification)
shap.initjs()
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][0],  # shap values for instance 0, class 0
    instance,
    matplotlib=True
)
