import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and prepare data
df = pd.read_csv('mcdonalds.csv')
print(f"Dataset shape: {df.shape}")

# Extract binary perception variables (first 11 columns)
X = df.iloc[:, :11]
X_binary = X.replace({'Yes': 1, 'No': 0})

# Standardize for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_binary)

# Find optimal number of clusters using BIC
n_components = range(2, 8)
bic_scores = []

for k in n_components:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))

# Select best k
best_k = n_components[np.argmin(bic_scores)]
print(f"Optimal clusters: {best_k}")

# Fit final model
gmm_final = GaussianMixture(n_components=best_k, random_state=42)
clusters = gmm_final.fit_predict(X_scaled)

# Add clusters to dataframe
df['cluster'] = clusters

# Analyze cluster characteristics
cluster_means = df.groupby('cluster')[X.columns].apply(
    lambda x: (x == 'Yes').mean()
).round(3)

print("\nCluster characteristics (proportion answering 'Yes'):")
print(cluster_means)

# Visualize results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Cluster sizes
cluster_counts = df['cluster'].value_counts().sort_index()
ax1.bar(cluster_counts.index, cluster_counts.values)
ax1.set_title('Cluster Sizes')
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Count')

# Heatmap of cluster profiles
sns.heatmap(cluster_means, annot=True, cmap='RdYlBu_r', ax=ax2)
ax2.set_title('Cluster Profiles')

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
ax3.set_title('Clusters in PCA Space')
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

# BIC plot
ax4.plot(n_components, bic_scores, 'bo-')
ax4.axvline(best_k, color='red', linestyle='--', alpha=0.7)
ax4.set_title('BIC Scores')
ax4.set_xlabel('Number of Components')
ax4.set_ylabel('BIC')

plt.tight_layout()
plt.show()

# Summary
print(f"\nClustering Summary:")
print(f"- {len(df)} customers segmented into {best_k} clusters")
print(f"- Cluster distribution: {dict(cluster_counts)}")

# Save results
df.to_csv('mcdonalds_clustered.csv', index=False)
print("Results saved to 'mcdonalds_clustered.csv'")
