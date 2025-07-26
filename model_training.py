# %% Import
from utils.helper_functions import Loader
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

# %% Load data
loader = Loader(r"data/framingham.csv")
loader.load_data()
loader.data.head()
loader.data.dtypes
loader.data.describe()

# %% Preprocess data
loader.preprocess()
# %% Split and Oversample
X_train, X_test, y_train, y_test = loader.get_data_split()
X_train, y_train = loader.oversample(X_train, y_train)
X_train = X_train.applymap(lambda x: int(x) if isinstance(x, bool) else x)

# %% Create Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
# %% Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# %% Train with early stopping
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
# %% Evaluate
loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}, Test AUC: {auc:.4f}")

# %% Visualize Training
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# %% Analyze Predictions
from sklearn.metrics import confusion_matrix, classification_report

y_pred = (model.predict(X_test) > 0.5).astype(int)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# %% save the model
model.save('model_name.keras')