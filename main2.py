import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from main import PCA_Scratch

# Load and preprocess data
data = load_breast_cancer()
X, y = data.data, data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test different numbers of components
n_components_range = range(1, 31)
cv_scores_pca = []
cv_scores_original = cross_val_score(LogisticRegression(random_state=42),
                                   X_scaled, y, cv=5).mean()

for n_comp in n_components_range:
    pca = PCA_Scratch(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    scores = cross_val_score(LogisticRegression(random_state=42),
                           X_pca, y, cv=5)
    cv_scores_pca.append(scores.mean())

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, cv_scores_pca, 'b-', label='PCA')
plt.axhline(y=cv_scores_original, color='r', linestyle='--',
            label='Original Data')
plt.xlabel('Number of Components')
plt.ylabel('Cross-validation Accuracy')
plt.title('PCA Components vs Model Performance')
plt.legend()
plt.grid(True)
plt.show()

print(f"Original data CV accuracy: {cv_scores_original:.4f}")
print("\nPCA CV accuracy with different components:")
for n_comp, score in zip(n_components_range, cv_scores_pca):
    if n_comp in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]:
        print(f"Components: {n_comp:2d}, Accuracy: {score:.4f}")
