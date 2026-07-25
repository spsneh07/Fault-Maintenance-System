import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Load data
df_phase3 = pd.read_csv('bearing_features_phase3.csv')
feature_cols = ['mean', 'std', 'variance', 'rms', 'peak', 'peak_to_peak', 'crest_factor', 'kurtosis', 'skewness']
features_df = df_phase3[feature_cols]
fault_labels = df_phase3['fault']

print("=== TASK 1: Correlation Matrix ===")
corr = features_df.corr()
print(corr)
# Find pairs > 0.85
pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        if abs(corr.iloc[i, j]) > 0.85:
            pairs.append((feature_cols[i], feature_cols[j], corr.iloc[i, j]))
print("Pairs with correlation > 0.85:")
for p in pairs:
    print(f"{p[0]} - {p[1]}: {p[2]:.4f}")

print("\n=== TASK 2: PCA ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df)

pca_full = PCA()
pca_full.fit(X_scaled)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)
print(f"Cumulative Variance: {cum_var}")
n_90 = np.argmax(cum_var >= 0.90) + 1
n_95 = np.argmax(cum_var >= 0.95) + 1
print(f"Components for 90% variance: {n_90}")
print(f"Components for 95% variance: {n_95}")

print("\n=== TASK 3: Biplot ===")
# We'll save the biplot as an image instead of displaying it
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
colors = {'OK': 'green', 'IR': 'red', 'OR': 'orange', 'RE': 'purple'}
for fault in colors:
    idx = fault_labels == fault
    plt.scatter(X_pca_2[idx, 0], X_pca_2[idx, 1], c=colors[fault], label=fault, alpha=0.7)
plt.xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA Biplot (PC1 vs PC2)')
plt.legend()
plt.grid(True)
plt.savefig('pca_biplot.png')
print("Saved biplot to pca_biplot.png")

print("\n=== TASK 4: PCA Loadings ===")
loadings = pd.DataFrame(pca_full.components_.T, columns=[f'PC{i+1}' for i in range(len(feature_cols))], index=feature_cols)
print("Loadings for PC1 and PC2:")
print(loadings[['PC1', 'PC2']])

print("\n=== TASK 5: RF with and without PCA ===")
X = features_df.values
le = LabelEncoder()
y = le.fit_transform(fault_labels)

# RF without PCA
pipe_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])
cv_no_pca = cross_val_score(pipe_no_pca, X, y, cv=5, scoring='accuracy')

# RF with PCA
pipe_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])
cv_pca = cross_val_score(pipe_pca, X, y, cv=5, scoring='accuracy')

print(f"RF without PCA: {cv_no_pca.mean():.4f} ± {cv_no_pca.std():.4f}")
print(f"RF with PCA (95% var): {cv_pca.mean():.4f} ± {cv_pca.std():.4f}")
