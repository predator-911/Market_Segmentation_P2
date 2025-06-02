import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv('mcdonalds.csv')
print(f"Dataset shape: {df.shape}")

# Prepare binary variables
X = df.iloc[:, :11].replace({'Yes': 1, 'No': 0})
X_scaled = StandardScaler().fit_transform(X)

# Elbow method analysis
k_range = range(2, 9)
inertia = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Find optimal k
best_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal k by silhouette score: {best_k}")

# Fit final model
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)
df['cluster'] = clusters

# Analyze clusters
cluster_summary = X.groupby(clusters).mean().round(3)
cluster_counts = pd.Series(clusters).value_counts().sort_index()

print(f"\nCluster sizes: {dict(cluster_counts)}")
print(f"\nCluster characteristics (proportion 'Yes'):")
print(cluster_summary)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Elbow plot
axes[0, 0].plot(k_range, inertia, 'bo-')
axes[0, 0].set_title('Elbow Method')
axes[0, 0].set_xlabel('Number of Clusters')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].grid(True, alpha=0.3)

# Silhouette scores
axes[0, 1].plot(k_range, silhouette_scores, 'ro-')
axes[0, 1].axvline(best_k, color='red', linestyle='--', alpha=0.7)
axes[0, 1].set_title('Silhouette Scores')
axes[0, 1].set_xlabel('Number of Clusters')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].grid(True, alpha=0.3)

# Cluster sizes
axes[1, 0].bar(cluster_counts.index, cluster_counts.values)
axes[1, 0].set_title('Cluster Sizes')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Count')

# Cluster heatmap
import seaborn as sns
sns.heatmap(cluster_summary, annot=True, cmap='RdYlBu_r', ax=axes[1, 1])
axes[1, 1].set_title('Cluster Profiles')

plt.tight_layout()
plt.show()

# Summary
print(f"\nFinal Summary:")
print(f"- {len(df)} customers segmented into {best_k} clusters")
print(f"- Silhouette score: {silhouette_scores[best_k-2]:.3f}")

# Save results
df.to_csv('mcdonalds_kmeans.csv', index=False)
print("Results saved to 'mcdonalds_kmeans.csv'")
