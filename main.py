import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class PCA_Scratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Calculate covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance

        return self

    def transform(self, X):
        # Center the data using mean from fit
        X_centered = X - self.mean

        # Project data onto principal components
        X_transformed = np.dot(X_centered, self.components)
        return X_transformed

    def fit_transform(self, X):
        # Convenience method to fit and transform in one step
        self.fit(X)
        return self.transform(X)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
n_components = 5
pca = PCA_Scratch(n_components=n_components)
# Now you can use either:
# Method 1: fit_transform in one step
X_train_pca = pca.fit_transform(X_train_scaled)
# Method 2: or fit and transform separately
# pca.fit(X_train_scaled)
# X_train_pca = pca.transform(X_train_scaled)

# Transform test data
X_test_pca = pca.transform(X_test_scaled)

# Train and evaluate models
# Original data
clf_original = LogisticRegression(random_state=42)
clf_original.fit(X_train_scaled, y_train)
y_pred_original = clf_original.predict(X_test_scaled)
acc_original = accuracy_score(y_test, y_pred_original)

# PCA transformed data
clf_pca = LogisticRegression(random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

# Plot explained variance ratio
plt.figure(figsize=(10, 5))
plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.tight_layout()
plt.show()

print(f"Original data accuracy: {acc_original:.4f}")
print(f"PCA transformed data accuracy: {acc_pca:.4f}")
print("\nExplained variance ratios:")
for i, ratio in enumerate(pca.explained_variance_ratio_, 1):
    print(f"PC{i}: {ratio:.4f}")
