# Gaussian Mixture Model Analysis - Local/Codespace Version
# Requirements: pandas, numpy, matplotlib, seaborn, scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path="mcdonalds.csv"):
    """
    Load the McDonald's dataset with comprehensive error handling
    """
    try:
        # Search in multiple common locations
        search_paths = [
            file_path,  # Current directory
            f"data/{file_path}",
            f"../data/{file_path}",
            f"datasets/{file_path}",
            f"../datasets/{file_path}",
            f"./data/{file_path}",
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"✅ Successfully loaded data from: {path}")
                print(f"📊 Dataset shape: {df.shape}")
                return df
        
        # If no file found, provide helpful guidance
        print(f"❌ Could not find '{file_path}' in any of these locations:")
        for path in search_paths:
            print(f"   - {path}")
        print("\n💡 Please ensure the McDonald's dataset is available.")
        print("   You can download it from Kaggle or other sources.")
        return None
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """
    Preprocess the McDonald's data for Gaussian Mixture Model clustering
    """
    if df is None:
        return None, None, None
    
    print("\n🔄 Preprocessing data...")
    print(f"Original data shape: {df.shape}")
    
    # Display basic info about the dataset
    print("\n📋 First few rows of the dataset:")
    print(df.head())
    
    # Select the 'Yes'/'No' columns for clustering (first 11 columns)
    MD_x = df.iloc[:, :11]
    print(f"\n🎯 Selected {MD_x.shape[1]} binary perception columns:")
    print(list(MD_x.columns))
    
    # Check unique values in each column to verify they're binary
    print("\n🔍 Unique values per column:")
    for col in MD_x.columns:
        unique_vals = MD_x[col].unique()
        print(f"   {col}: {unique_vals}")
    
    # Convert 'Yes' to 1 and 'No' to 0
    MD_x_encoded = MD_x.apply(lambda x: x.map({'Yes': 1, 'No': 0}))
    
    # Check for any unmapped values
    if MD_x_encoded.isnull().any().any():
        print("\n⚠️  Warning: Some values couldn't be mapped to 0/1:")
        for col in MD_x.columns:
            if MD_x_encoded[col].isnull().any():
                unmapped = MD_x[col][MD_x_encoded[col].isnull()].unique()
                print(f"   {col}: {unmapped}")
        
        # Fill NaN values with 0 (assuming 'No' for unmapped values)
        MD_x_encoded = MD_x_encoded.fillna(0)
        print("   🔧 Filled unmapped values with 0")
    
    # Convert to integer type for consistency
    MD_x_encoded = MD_x_encoded.astype(int)
    
    # Optional: Standardize the data (though not always necessary for GMM)
    scaler = StandardScaler()
    MD_scaled = scaler.fit_transform(MD_x_encoded)
    
    print(f"✅ Preprocessing complete!")
    print(f"   - Encoded data shape: {MD_x_encoded.shape}")
    print(f"   - Data range: {MD_x_encoded.min().min()} to {MD_x_encoded.max().max()}")
    
    return MD_x_encoded, MD_scaled, scaler

def perform_gmm_analysis(MD_x, n_components_range=None):
    """
    Perform Gaussian Mixture Model analysis with AIC and BIC evaluation
    """
    if MD_x is None:
        return None, None, None
    
    if n_components_range is None:
        n_components_range = range(2, 9)
    
    print(f"\n📈 Performing GMM analysis for k = {min(n_components_range)} to {max(n_components_range)}...")
    
    aic_scores = []
    bic_scores = []
    log_likelihoods = []
    silhouette_scores = []
    models = {}
    
    for n_components in n_components_range:
        print(f"   Testing k = {n_components}...", end=" ")
        
        # Fit Gaussian Mixture Model
        gm = GaussianMixture(n_components=n_components, random_state=1234, n_init=10)
        gm.fit(MD_x)
        
        # Store model
        models[n_components] = gm
        
        # Calculate information criteria
        aic = gm.aic(MD_x)
        bic = gm.bic(MD_x)
        log_likelihood = gm.score(MD_x) * len(MD_x)  # Total log-likelihood
        
        aic_scores.append(aic)
        bic_scores.append(bic)
        log_likelihoods.append(log_likelihood)
        
        # Calculate silhouette score
        labels = gm.predict(MD_x)
        if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
            sil_score = silhouette_score(MD_x, labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(0)
        
        print(f"AIC: {aic:.2f}, BIC: {bic:.2f}, Silhouette: {silhouette_scores[-1]:.3f}")
    
    return (list(n_components_range), aic_scores, bic_scores, 
            log_likelihoods, silhouette_scores, models)

def create_comprehensive_plots(n_components_range, aic_scores, bic_scores, 
                             log_likelihoods, silhouette_scores, save_plots=True):
    """
    Create comprehensive visualization plots for GMM analysis
    """
    print("\n📊 Creating visualization plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Gaussian Mixture Model Analysis for McDonald\'s Customer Segmentation', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Information Criteria (AIC & BIC)
    axes[0, 0].plot(n_components_range, aic_scores, marker='o', linewidth=2, 
                   markersize=8, color='#E74C3C', label='AIC')
    axes[0, 0].plot(n_components_range, bic_scores, marker='s', linewidth=2, 
                   markersize=8, color='#3498DB', label='BIC')
    axes[0, 0].set_xlabel('Number of Components')
    axes[0, 0].set_ylabel('Information Criteria Value')
    axes[0, 0].set_title('Information Criteria (AIC & BIC)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(n_components_range)
    axes[0, 0].legend()
    
    # Add annotations for minimum values
    min_aic_idx = np.argmin(aic_scores)
    min_bic_idx = np.argmin(bic_scores)
    axes[0, 0].annotate(f'Min AIC\nk={n_components_range[min_aic_idx]}', 
                       xy=(n_components_range[min_aic_idx], aic_scores[min_aic_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#E74C3C', alpha=0.3),
                       arrowprops=dict(arrowstyle='->', color='#E74C3C'))
    axes[0, 0].annotate(f'Min BIC\nk={n_components_range[min_bic_idx]}', 
                       xy=(n_components_range[min_bic_idx], bic_scores[min_bic_idx]),
                       xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#3498DB', alpha=0.3),
                       arrowprops=dict(arrowstyle='->', color='#3498DB'))
    
    # Plot 2: Log-Likelihood
    axes[0, 1].plot(n_components_range, log_likelihoods, marker='^', linewidth=2, 
                   markersize=8, color='#2ECC71')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Log-Likelihood')
    axes[0, 1].set_title('Log-Likelihood vs Number of Components')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(n_components_range)
    
    # Plot 3: Silhouette Scores
    axes[1, 0].bar(n_components_range, silhouette_scores, color='#9B59B6', alpha=0.7)
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('Silhouette Score')
    axes[1, 0].set_title('Silhouette Score Analysis')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_xticks(n_components_range)
    
    # Add value labels on bars
    for i, (k, score) in enumerate(zip(n_components_range, silhouette_scores)):
        axes[1, 0].text(k, score + 0.005, f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Combined Criteria Comparison (Normalized)
    # Normalize scores to 0-1 range for comparison
    aic_norm = (np.array(aic_scores) - min(aic_scores)) / (max(aic_scores) - min(aic_scores))
    bic_norm = (np.array(bic_scores) - min(bic_scores)) / (max(bic_scores) - min(bic_scores))
    # Invert silhouette for consistency (lower is better for AIC/BIC, higher is better for silhouette)
    sil_norm = 1 - np.array(silhouette_scores) / max(silhouette_scores) if max(silhouette_scores) > 0 else np.zeros_like(silhouette_scores)
    
    axes[1, 1].plot(n_components_range, aic_norm, marker='o', label='AIC (norm)', color='#E74C3C')
    axes[1, 1].plot(n_components_range, bic_norm, marker='s', label='BIC (norm)', color='#3498DB')
    axes[1, 1].plot(n_components_range, sil_norm, marker='^', label='1-Silhouette (norm)', color='#9B59B6')
    axes[1, 1].set_xlabel('Number of Components')
    axes[1, 1].set_ylabel('Normalized Score (0=Best, 1=Worst)')
    axes[1, 1].set_title('Normalized Criteria Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(n_components_range)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_plots:
        filename = 'gmm_information_criteria_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Plots saved as '{filename}'")
    
    # Display plot if possible
    try:
        plt.show()
    except:
        print("📊 Plots created but display not available in this environment")
    
    return fig

def recommend_optimal_components(n_components_range, aic_scores, bic_scores, silhouette_scores):
    """
    Provide recommendations for optimal number of components based on multiple criteria
    """
    print("\n🎯 OPTIMAL COMPONENT RECOMMENDATIONS:")
    print("=" * 50)
    
    # Find optimal values for each criterion
    min_aic_idx = np.argmin(aic_scores)
    min_bic_idx = np.argmin(bic_scores)
    max_sil_idx = np.argmax(silhouette_scores)
    
    optimal_aic = n_components_range[min_aic_idx]
    optimal_bic = n_components_range[min_bic_idx]
    optimal_sil = n_components_range[max_sil_idx]
    
    print(f"📊 Individual Criterion Recommendations:")
    print(f"   AIC: k = {optimal_aic} (AIC = {aic_scores[min_aic_idx]:.2f})")
    print(f"   BIC: k = {optimal_bic} (BIC = {bic_scores[min_bic_idx]:.2f})")
    print(f"   Silhouette: k = {optimal_sil} (Score = {silhouette_scores[max_sil_idx]:.3f})")
    
    # Show all scores in table format
    print(f"\n📋 Complete Results Table:")
    print(f"{'k':<3} {'AIC':<10} {'BIC':<10} {'Silhouette':<12} {'Rank Sum':<8}")
    print("-" * 45)
    
    # Calculate ranking for each metric
    aic_ranks = np.argsort(aic_scores) + 1
    bic_ranks = np.argsort(bic_scores) + 1
    sil_ranks = np.argsort(silhouette_scores)[::-1] + 1  # Reverse for silhouette (higher is better)
    
    rank_sums = aic_ranks + bic_ranks + sil_ranks
    
    for i, k in enumerate(n_components_range):
        print(f"{k:<3} {aic_scores[i]:<10.2f} {bic_scores[i]:<10.2f} "
              f"{silhouette_scores[i]:<12.3f} {rank_sums[i]:<8}")
    
    # Find best overall (lowest rank sum)
    best_overall_idx = np.argmin(rank_sums)
    best_overall_k = n_components_range[best_overall_idx]
    
    print(f"\n💡 OVERALL RECOMMENDATIONS:")
    print(f"   🥇 Best Overall: k = {best_overall_k} (lowest rank sum: {rank_sums[best_overall_idx]})")
    
    # Show top 3 by rank sum
    sorted_indices = np.argsort(rank_sums)
    print(f"   📊 Top 3 by Combined Ranking:")
    for i, idx in enumerate(sorted_indices[:3]):
        k = n_components_range[idx]
        print(f"      {i+1}. k = {k} (rank sum: {rank_sums[idx]})")
    
    # BIC is often preferred for model selection
    print(f"\n🎯 RECOMMENDED CHOICE:")
    if optimal_bic == best_overall_k:
        print(f"   k = {optimal_bic} (agrees with both BIC and overall ranking)")
    else:
        print(f"   Consider k = {optimal_bic} (BIC preference) or k = {best_overall_k} (best overall)")
    
    return best_overall_k

def analyze_optimal_model(df, MD_x, optimal_k, models):
    """
    Analyze the characteristics of the optimal GMM model
    """
    print(f"\n🔍 Analyzing optimal GMM model with k = {optimal_k}...")
    
    # Get the optimal model
    optimal_model = models[optimal_k]
    
    # Get cluster assignments
    cluster_labels = optimal_model.predict(MD_x)
    cluster_probs = optimal_model.predict_proba(MD_x)
    
    # Add to dataframe
    df['GMM_Cluster'] = cluster_labels
    df['GMM_Max_Probability'] = np.max(cluster_probs, axis=1)
    
    # Analyze cluster distribution
    print(f"📊 Cluster distribution:")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   Cluster {cluster}: {count} customers ({percentage:.1f}%)")
    
    # Analyze cluster characteristics
    print(f"\n🎯 Cluster characteristics (average response rates):")
    cluster_characteristics = []
    
    for i in range(optimal_k):
        mask = cluster_labels == i
        cluster_mean = MD_x[mask].mean()
        cluster_characteristics.append(cluster_mean)
        
        print(f"\n   📍 Cluster {i} ({sum(mask)} customers):")
        print(f"      Average assignment probability: {cluster_probs[mask, i].mean():.3f}")
        
        # Show top characteristics for this cluster
        sorted_features = cluster_mean.sort_values(ascending=False)
        print(f"      Top characteristics:")
        for feature, value in sorted_features.head(5).items():
            print(f"         {feature}: {value:.2f} ({value*100:.0f}% Yes)")
    
    # Create cluster characteristics dataframe
    cluster_df = pd.DataFrame(cluster_characteristics, 
                             columns=MD_x.columns,
                             index=[f'Cluster {i}' for i in range(optimal_k)])
    
    return optimal_model, df, cluster_df

def create_cluster_visualization(MD_x, cluster_labels, cluster_df, optimal_k):
    """
    Create visualizations for cluster analysis
    """
    print("\n📊 Creating cluster visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'GMM Cluster Analysis (k={optimal_k})', fontsize=16, fontweight='bold')
    
    # Plot 1: Cluster size distribution
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
    
    bars = axes[0, 0].bar(range(optimal_k), cluster_counts, color=colors)
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('Number of Customers')
    axes[0, 0].set_title('Cluster Size Distribution')
    axes[0, 0].set_xticks(range(optimal_k))
    
    # Add value labels on bars
    for bar, count in zip(bars, cluster_counts):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({count/len(cluster_labels)*100:.1f}%)',
                       ha='center', va='bottom')
    
    # Plot 2: Cluster characteristics heatmap
    sns.heatmap(cluster_df.T, annot=True, cmap='RdYlBu_r', ax=axes[0, 1],
                cbar_kws={'label': 'Average Response (0=No, 1=Yes)'}, fmt='.2f')
    axes[0, 1].set_title('Cluster Characteristics Heatmap')
    axes[0, 1].set_xlabel('Cluster')
    
    # Plot 3: PCA visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(MD_x)
    
    for i in range(optimal_k):
        mask = cluster_labels == i
        axes[1, 0].scatter(pca_result[mask, 0], pca_result[mask, 1], 
                          c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=50)
    
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1, 0].set_title(f'Clusters in PCA Space\n(Total explained variance: {sum(pca.explained_variance_ratio_):.2%})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Feature importance by cluster variance
    feature_variance = cluster_df.var(axis=0).sort_values(ascending=True)
    axes[1, 1].barh(range(len(feature_variance)), feature_variance.values, color='skyblue')
    axes[1, 1].set_yticks(range(len(feature_variance)))
    axes[1, 1].set_yticklabels(feature_variance.index, fontsize=10)
    axes[1, 1].set_xlabel('Variance Across Clusters')
    axes[1, 1].set_title('Feature Discriminatory Power\n(Higher variance = better cluster separation)')
    axes[1, 1].grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = 'gmm_cluster_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Cluster visualization saved as '{filename}'")
    
    try:
        plt.show()
    except:
        print("📊 Cluster plots created but display not available in this environment")

def main():
    """
    Main analysis function
    """
    print("🍟 Gaussian Mixture Model Analysis for McDonald's Customer Segmentation")
    print("=" * 75)
    
    # Load and preprocess data
    df = load_data("mcdonalds.csv")
    MD_x_encoded, MD_scaled, scaler = preprocess_data(df)
    
    if MD_x_encoded is None:
        print("❌ Cannot proceed without data. Please check your dataset.")
        return None
    
    # Perform GMM analysis
    results = perform_gmm_analysis(MD_x_encoded)
    if results[0] is None:
        print("❌ GMM analysis failed.")
        return None
    
    n_components_range, aic_scores, bic_scores, log_likelihoods, silhouette_scores, models = results
    
    # Create information criteria plots
    fig = create_comprehensive_plots(n_components_range, aic_scores, bic_scores, 
                                   log_likelihoods, silhouette_scores)
    
    # Recommend optimal number of components
    optimal_k = recommend_optimal_components(n_components_range, aic_scores, 
                                           bic_scores, silhouette_scores)
    
    # Analyze optimal model
    optimal_model, df_with_clusters, cluster_df = analyze_optimal_model(
        df, MD_x_encoded, optimal_k, models)
    
    # Create cluster visualizations
    create_cluster_visualization(MD_x_encoded, df_with_clusters['GMM_Cluster'], 
                               cluster_df, optimal_k)
    
    # Save results
    output_file = 'mcdonalds_gmm_results.csv'
    df_with_clusters.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to '{output_file}'")
    
    # Save cluster characteristics
    cluster_file = 'gmm_cluster_characteristics.csv'
    cluster_df.to_csv(cluster_file)
    print(f"💾 Cluster characteristics saved to '{cluster_file}'")
    
    print("\n✅ GMM Analysis completed successfully!")
    
    return {
        'dataframe': df_with_clusters,
        'model': optimal_model,
        'cluster_characteristics': cluster_df,
        'n_components_range': n_components_range,
        'aic_scores': aic_scores,
        'bic_scores': bic_scores,
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k
    }

if __name__ == "__main__":
    try:
        results = main()
        if results:
            print(f"\n🎯 Final Results Summary:")
            print(f"   - Optimal components: {results['optimal_k']}")
            print(f"   - Model type: Gaussian Mixture Model")
            print(f"   - Data points: {len(results['dataframe'])}")
            print(f"   - Features used: {results['dataframe'].iloc[:, :11].shape[1]}")
            print(f"   - Best AIC: {min(results['aic_scores']):.2f}")
            print(f"   - Best BIC: {min(results['bic_scores']):.2f}")
            print(f"   - Best Silhouette: {max(results['silhouette_scores']):.3f}")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure all required packages are installed:")
        print("   pip install pandas numpy matplotlib seaborn scikit-learn")
        print("2. Make sure the McDonald's dataset is in the correct location")
        print("3. Check that the dataset has the expected format (11 binary columns)")
        print("4. Verify that the data contains 'Yes'/'No' values in the first 11 columns")