import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import to_categorical
import seaborn as sns

# ─── STEP 1: LOAD DATA ─────────────────────────────────────────────────────
df = pd.read_csv('bearing_features_phase3.csv')

# Features the model receives as input
feature_cols = ['mean', 'std', 'variance', 'rms', 'peak',
                'peak_to_peak', 'crest_factor', 'kurtosis', 'skewness']

X = df[feature_cols].values   # shape: (300, 9)
y = df['fault'].values          # shape: (300,)  e.g. ['OK','IR','OR','RE']

# ─── STEP 2: ENCODE LABELS ─────────────────────────────────────────────────
le = LabelEncoder()
y_int = le.fit_transform(y)     # OK→0, IR→1, OR→2, RE→3  (integer labels)
y_ohe = to_categorical(y_int)   # shape: (300, 4)  → one-hot vectors

num_classes = y_ohe.shape[1]   # = 4
print(f"Classes: {le.classes_}")

# ─── STEP 3: TRAIN/TEST SPLIT ──────────────────────────────────────────────
# stratify=y_int ensures each fault class has equal representation in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y_ohe, test_size=0.2, random_state=42, stratify=y_int
)
# X_train: (240, 9)   X_test: (60, 9)

# ─── STEP 4: SCALE FEATURES ────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # compute μ,σ from TRAIN only
X_test  = scaler.transform(X_test)       # apply same μ,σ to TEST (no leakage)

# ─── STEP 5: BUILD ANN ARCHITECTURE ────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=(9,)),               # 9 features per sample

    layers.Dense(64, activation='relu'),    # hidden layer 1: 9×64 weights + 64 biases
    layers.BatchNormalization(),              # stabilises training, speeds convergence
    layers.Dropout(0.3),                    # randomly zeros 30% of neurons → regularisation

    layers.Dense(32, activation='relu'),    # hidden layer 2: 64×32 weights + 32 biases
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(num_classes,                # output: 4 neurons (one per fault class)
                 activation='softmax')      # converts scores → probabilities summing to 1
], name='bearing_ann')

model.summary()   # prints layer-by-layer parameter count

# ─── STEP 6: COMPILE ────────────────────────────────────────────────────────
model.compile(
    optimizer='adam',                        # adaptive learning rate
    loss='categorical_crossentropy',         # multi-class cross entropy
    metrics=['accuracy']
)

# ─── STEP 7: TRAIN ──────────────────────────────────────────────────────────
# Early stopping: stop if val_loss doesn't improve for 15 epochs in a row
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,          # 32 samples per gradient update
    validation_split=0.2,   # uses 20% of training set as validation
    callbacks=[early_stop],
    verbose=1
)

# ─── STEP 8: EVALUATE ───────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}  |  Test Loss: {loss:.4f}")

y_pred_proba = model.predict(X_test)            # shape: (60, 4)
y_pred_int   = np.argmax(y_pred_proba, axis=1) # pick class with highest probability
y_true_int   = np.argmax(y_test, axis=1)

print(classification_report(y_true_int, y_pred_int,
      target_names=le.classes_))

# ─── STEP 9: PLOTS ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy')
axes[0].legend()

cm = confusion_matrix(y_true_int, y_pred_int)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[1].set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('ann_results.png', dpi=150)
print('ann_results.png saved successfully.')
# plt.show()  # Commented out to avoid blocking terminal execution
