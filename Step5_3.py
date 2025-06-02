import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv('mcdonalds.csv')
print(f"Dataset shape: {df.shape}")

# Prepare binary variables
X = df.iloc[:, :11].replace({'Yes': 1, 'No': 0})
X_scaled = StandardScaler().fit_transform(X)

# GMM analysis with different numbers of components
k_range = range(2, 9)
aic_scores = []
bic_scores = []
silhouette_scores = []
models = {}

for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
    gmm.fit(X_scaled)
    
    models[k] = gmm
    aic_scores.append(gmm.aic(X_scaled))
    bic_scores.append(gmm.bic(X_scaled))
    
    labels = gmm.predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Find optimal k
best_k_aic = k_range[np.argmin(aic_scores)]
best_k_bic = k_range[np.argmin(bic_scores)]
best_k_sil = k_range[np.argmax(silhouette_scores)]

print(f"Optimal k by AIC: {best_k_aic}")
print(f"Optimal k by BIC: {best_k_bic}")
print(f"Optimal k by silhouette: {best_k_sil}")

# Use BIC recommendation (most conservative)
best_k = best_k_bic
print(f"Final choice: {best_k} components")

# Fit final model
gmm_final = models[best_k]
clusters = gmm_final.predict(X_scaled)
probabilities = gmm_final.predict_proba(X_scaled)

df['cluster'] = clusters
df['max_probability'] = np.max(probabilities, axis=1)

# Analyze clusters
cluster_summary = X.groupby(clusters).mean().round(3)
cluster_counts = pd.Series(clusters).value_counts().sort_index()

print(f"\nCluster sizes: {dict(cluster_counts)}")
print(f"\nCluster characteristics (proportion 'Yes'):")
print(cluster_summary)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Information criteria
axes[0, 0].plot(k_range, aic_scores, 'bo-', label='AIC')
axes[0, 0].plot(k_range, bic_scores, 'go-', label='BIC')
axes[0, 0].axvline(best_k, color='red', linestyle='--', alpha=0.7)
axes[0, 0].set_title('Information Criteria')
axes[0, 0].set_xlabel('Number of Components')
axes[0, 0].set_ylabel('Score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Silhouette scores
axes[0, 1].plot(k_range, silhouette_scores, 'ro-')
axes[0, 1].axvline(best_k, color='red', linestyle='--', alpha=0.7)
axes[0, 1].set_title('Silhouette Scores')
axes[0, 1].set_xlabel('Number of Components')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].grid(True, alpha=0.3)

# Cluster sizes
axes[1, 0].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
axes[1, 0].set_title('Cluster Sizes')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Count')

# Add count labels
for i, count in enumerate(cluster_counts.values):
    axes[1, 0].text(i, count + 5, str(count), ha='center', va='bottom')

# Cluster heatmap
sns.heatmap(cluster_summary, annot=True, cmap='RdYlBu_r', ax=axes[1, 1], fmt='.2f')
axes[1, 1].set_title('Cluster Profiles (GMM)')
axes[1, 1].set_xlabel('Cluster')

plt.tight_layout()
plt.show()

# Model comparison table
print(f"\nModel Selection Results:")
print(f"{'k':<3} {'AIC':<8} {'BIC':<8} {'Silhouette':<10}")
print("-" * 35)
for i, k in enumerate(k_range):
    print(f"{k:<3} {aic_scores[i]:<8.1f} {bic_scores[i]:<8.1f} {silhouette_scores[i]:<10.3f}")

# Cluster interpretation
print(f"\nCluster Interpretation:")
for i in range(best_k):
    mask = clusters == i
    size = sum(mask)
    avg_prob = probabilities[mask, i].mean()
    
    print(f"\nCluster {i}: {size} customers ({size/len(df)*100:.1f}%)")
    print(f"  Average assignment probability: {avg_prob:.3f}")
    
    # Top characteristics
    cluster_means = cluster_summary.iloc[i].sort_values(ascending=False)
    print(f"  Top characteristics:")
    for attr, value in cluster_means.head(3).items():
        print(f"    {attr}: {value:.2f} ({value*100:.0f}% Yes)")

# Summary
print(f"\nFinal Summary:")
print(f"- {len(df)} customers segmented into {best_k} components")
print(f"- Best AIC: {min(aic_scores):.1f}")
print(f"- Best BIC: {min(bic_scores):.1f}")
print(f"- Best silhouette: {max(silhouette_scores):.3f}")
print(f"- Average assignment certainty: {df['max_probability'].mean():.3f}")

# Save results
df.to_csv('mcdonalds_gmm.csv', index=False)
cluster_summary.to_csv('gmm_cluster_profiles.csv')
print("\nResults saved to 'mcdonalds_gmm.csv' and 'gmm_cluster_profiles.csv'")
