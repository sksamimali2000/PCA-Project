# Principal Component Analysis (PCA) and Dimensionality Reduction

## 📊 1. Dimensionality Reduction

In many datasets, there are often too many features based on which conclusions are drawn.  
Higher feature dimensions make visualization and processing harder, especially if many features are correlated and redundant.

**Dimensionality Reduction** is the process of reducing the number of random variables under consideration by obtaining a set of principal variables.

### Example Visualization:
<img width="551" height="539" alt="image" src="https://github.com/user-attachments/assets/a53c7c2c-ac94-49c3-aed7-0ee128b2bf3b" />

---

## ✅ 2. Advantages of Dimensionality Reduction

1. Reduces time and storage space.
2. Removes multi-collinearity improving ML model performance.
3. Easier data visualization in low dimensions (2D/3D).

---

## ⚠️ 3. Disadvantages of Dimensionality Reduction

1. Potential data loss.
2. PCA finds linear correlations only.
3. Ineffective if mean & covariance are insufficient.
4. Selecting the number of principal components (k) is tricky.

---

## 🔍 4. Principal Component Analysis (PCA)

PCA transforms original variables into a new set of variables (principal components):  
- First PC explains max variance.
- Each succeeding PC is orthogonal to previous PCs.

For a 2D dataset, there are exactly two PCs, making further reduction meaningless.

---

## ⚡ 5. PCA vs Linear Regression

| PCA | Linear Regression |
|-----|-------------------|
| Projects data to minimize squared projection error. | Fits a line to predict a dependent variable. |
| Not for prediction. | Used for prediction tasks. |

For a great visualization, visit:  
[Principal Component Analysis Visualization](http://setosa.io/ev/principal-component-analysis/)

![PCA vs Regression](3pca.bmp)

---

## 🧱 6. PCA Algorithm Steps

1. Preprocess data (scaling/normalization).
2. Calculate covariance matrix.
3. Compute eigenvalues & eigenvectors.
4. Select top k eigenvectors (highest variance).
5. Transform dataset into lower dimensions.

---

## ✅ PCA Implementation Code

```python
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
mean_vec1 = np.array([0,0,0])
cov_mat1 = np.eye(3)
class1 = np.random.multivariate_normal(mean_vec1, cov_mat1, 20)

mean_vec2 = np.array([1,1,1])
cov_mat2 = np.eye(3)
class2 = np.random.multivariate_normal(mean_vec2, cov_mat2, 20)

all_data = np.concatenate((class1, class2))

# Manual PCA Implementation
cov_mat = np.cov(all_data.T)
eig_val, eig_vec = np.linalg.eig(cov_mat)

# Select top-2 components
matrix_eig = np.array([eig_vec[:,0], eig_vec[:,1]])
transformed = matrix_eig.dot(all_data.T).T

# Visualization
plt.plot(transformed[:20, 0], transformed[:20, 1], 'o')
plt.plot(transformed[20:, 0], transformed[20:, 1], '^')
plt.show()

# PCA Using sklearn
pca = PCA(n_components=2)
skl_transformed = pca.fit_transform(all_data)
plt.plot(skl_transformed[:20, 0], skl_transformed[:20, 1], 'o')
plt.plot(skl_transformed[20:, 0], skl_transformed[20:, 1], '^')
plt.show()
```


🎯 How to Select k (Number of Components)

Goal: Small k that explains max variance.

Example using explained variance ratio:
```Python
pca = PCA()
pca.fit(class1)
total_variance = 0
k = 0
while total_variance < 0.9:
    total_variance += pca.explained_variance_ratio_[k]
    k += 1

print(f"Optimal k for 90% variance: {k}")

pca = PCA(n_components=k)
skl_transformed = pca.fit_transform(class1)
plt.plot(skl_transformed, 'o', color='m')
plt.show()
```

🔧 PCA Approximation Example
```Python
X_approx = pca.inverse_transform(skl_transformed)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(X_approx[:, 0], X_approx[:, 1], X_approx[:, 2], '^')
plt.show()
```

✅ Conclusion

PCA helps reduce dimensionality and extract meaningful features, facilitating visualization and improving model performance in most cases.
However, careful selection of k and understanding PCA assumptions is important for optimal results.
