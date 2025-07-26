# %% Import
import pandas as pd
import numpy as np
from keras.models import load_model

# %% Load trained Keras model
model = load_model("model_name.keras")

# %% Load your data 
from utils.helper_functions import Loader
loader = Loader(r"data/framingham.csv")
loader.load_data()
loader.preprocess()

# %% Split
X_train, X_test, y_train, y_test = loader.get_data_split()

# %% Running Lime for local Interpretation
from lime.lime_tabular import LimeTabularExplainer
X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train

explainer = LimeTabularExplainer(
    training_data=X_train_np,
    feature_names=X_train.columns.tolist(),
    class_names=['Healthy future', 'Risky Future'],
    mode='classification',
    discretize_continuous=True
)

# %% Explain
X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

def predict_fn(x):
    preds = model.predict(x)
    return np.concatenate([1 - preds, preds], axis=1)

instance_index = 5
instance = X_test_np[instance_index]

# Explain the selected instance
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=predict_fn,
    num_features=15
)

# Show explanation
exp.show_in_notebook()
# Or to print feature importances
for feature, weight in exp.as_list():
    print(f"{feature}: {weight:.4f}")

# %%
