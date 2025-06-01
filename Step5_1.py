# McDonald's Data Analysis - Local/Codespace Version
# Requirements: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

def load_data(file_path="mcdonalds.csv"):
    """
    Load the McDonald's dataset with error handling
    """
    try:
        # Try to load from current directory first
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✓ Successfully loaded data from {file_path}")
            return df
        
        # Try common data directories
        data_paths = [
            f"data/{file_path}",
            f"../data/{file_path}",
            f"datasets/{file_path}",
            f"../datasets/{file_path}"
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"✓ Successfully loaded data from {path}")
                return df
        
        # If file not found, provide instructions
        print(f"❌ Could not find {file_path} in current directory or common data folders.")
        print("Please ensure the McDonald's dataset is available in one of these locations:")
        print("- Current directory")
        print("- data/ folder")
        print("- datasets/ folder")
        print("\nYou can download the dataset from: https://www.kaggle.com/datasets/mcdonalds/nutrition-facts")
        return None
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """
    Preprocess the McDonald's data
    """
    print("📊 Data shape:", df.shape)
    print("\n📋 First few rows:")
    print(df.head())
    
    # Select the 'Yes'/'No' columns for clustering (assuming first 11 columns)
    MD_x = df.iloc[:, :11]
    print(f"\n🔍 Selected {MD_x.shape[1]} binary perception columns for analysis")
    
    # Convert 'Yes' to 1 and 'No' to 0
    MD_x_encoded = MD_x.apply(lambda x: x.map({'Yes': 1, 'No': 0}))
    
    # Check for any non-binary values that couldn't be mapped
    if MD_x_encoded.isnull().any().any():
        print("⚠️ Warning: Some values couldn't be mapped to 0/1. Check unique values:")
        for col in MD_x.columns:
            unique_vals = MD_x[col].unique()
            if len(unique_vals) > 2:
                print(f"Column '{col}': {unique_vals}")
    
    # Standardize the data
    scaler = StandardScaler()
    MD_scaled = scaler.fit_transform(MD_x_encoded)
    
    return MD_x_encoded, MD_scaled, scaler

def process_like_variable(df):
    """
    Process the 'Like' variable if it exists
    """
    if 'Like' not in df.columns:
        print("⚠️ 'Like' column not found in dataset. Skipping Like variable processing.")
        return None
    
    print("🎯 Unique values in 'Like' column:")
    print(df['Like'].unique())
    
    # Define mapping for Like variable
    like_mapping = {
        'I hate it!-5': -5,
        '-4': -4,
        '-3': -3,
        '-2': -2,
        '-1': -1,
        '0': 0,
        '+1': 1,
        '+2': 2,
        '+3': 3,
        '+4': 4,
        'I love it!+5': 5
    }
    
    df['Like_numeric'] = df['Like'].map(like_mapping)
    
    print("\n📈 Like variable mapping sample:")
    print(df[['Like', 'Like_numeric']].head())
    print(f"Like_numeric data type: {df['Like_numeric'].dtype}")
    
    return df['Like_numeric']

def evaluate_gmm_components(independent_variables):
    """
    Evaluate different numbers of GMM components using BIC
    """
    print("🔍 Evaluating optimal number of GMM components...")
    
    # Define the range for the number of components
    n_components_range = range(2, 9)
    bic_scores = []
    
    for n_components in n_components_range:
        gm = GaussianMixture(n_components=n_components, random_state=1234, n_init=10)
        gm.fit(independent_variables)
        bic_scores.append(gm.bic(independent_variables))
    
    # Display the BIC scores
    print("\n📊 BIC scores for Gaussian Mixture Models:")
    for k, bic in zip(n_components_range, bic_scores):
        print(f"k={k}: BIC = {bic:.2f}")
    
    # Find optimal number of components (lowest BIC)
    optimal_k = n_components_range[np.argmin(bic_scores)]
    print(f"\n✓ Optimal number of components: k={optimal_k} (lowest BIC: {min(bic_scores):.2f})")
    
    return optimal_k, bic_scores

def fit_optimal_gmm(independent_variables, optimal_k):
    """
    Fit GMM with optimal number of components and assign clusters
    """
    print(f"\n🎯 Fitting GMM with k={optimal_k} components...")
    
    gm_optimal = GaussianMixture(n_components=optimal_k, random_state=1234, n_init=10)
    gm_optimal.fit(independent_variables)
    
    # Get cluster assignments
    cluster_assignments = gm_optimal.predict(independent_variables)
    
    print(f"\n📈 Distribution of data points across {optimal_k} GMM clusters:")
    cluster_counts = pd.Series(cluster_assignments).value_counts().sort_index()
    print(cluster_counts)
    
    return gm_optimal, cluster_assignments

def visualize_results(independent_variables, cluster_assignments, optimal_k):
    """
    Create visualizations of the clustering results
    """
    print("\n📊 Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('McDonald\'s Customer Segmentation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cluster distribution
    axes[0, 0].bar(range(optimal_k), pd.Series(cluster_assignments).value_counts().sort_index())
    axes[0, 0].set_title('Distribution of Customers Across Clusters')
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('Number of Customers')
    
    # 2. Heatmap of cluster means
    cluster_means = []
    for i in range(optimal_k):
        mask = cluster_assignments == i
        cluster_mean = independent_variables[mask].mean(axis=0)
        cluster_means.append(cluster_mean)
    
    cluster_means_df = pd.DataFrame(cluster_means, 
                                   columns=independent_variables.columns,
                                   index=[f'Cluster {i}' for i in range(optimal_k)])
    
    sns.heatmap(cluster_means_df.T, annot=True, cmap='RdYlBu_r', ax=axes[0, 1], 
                cbar_kws={'label': 'Average Response (0=No, 1=Yes)'})
    axes[0, 1].set_title('Cluster Characteristics Heatmap')
    
    # 3. Principal component visualization (if possible)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(independent_variables)
    
    scatter = axes[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                c=cluster_assignments, cmap='viridis', alpha=0.6)
    axes[1, 0].set_title(f'Clusters in PCA Space\n(Explained variance: {sum(pca.explained_variance_ratio_):.2%})')
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[1, 0], label='Cluster')
    
    # 4. Silhouette score
    if len(np.unique(cluster_assignments)) > 1:
        silhouette_avg = silhouette_score(independent_variables, cluster_assignments)
        axes[1, 1].text(0.5, 0.5, f'Silhouette Score\n{silhouette_avg:.3f}', 
                       ha='center', va='center', fontsize=20, fontweight='bold',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Model Quality Metric')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('mcdonalds_clustering_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 Visualization saved as 'mcdonalds_clustering_analysis.png'")
    
    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        print("📊 Plot created but display not available in this environment")
    
    return cluster_means_df

def main():
    """
    Main analysis function
    """
    print("🍟 McDonald's Customer Segmentation Analysis")
    print("=" * 50)
    
    # 1. Load data
    df = load_data("mcdonalds.csv")
    if df is None:
        return
    
    # 2. Preprocess data
    independent_variables, scaled_data, scaler = preprocess_data(df)
    
    # 3. Process Like variable (if exists)
    dependent_variable = process_like_variable(df)
    
    # 4. Evaluate GMM components
    optimal_k, bic_scores = evaluate_gmm_components(independent_variables)
    
    # 5. Fit optimal GMM
    gm_optimal, cluster_assignments = fit_optimal_gmm(independent_variables, optimal_k)
    
    # 6. Add cluster assignments to original dataframe
    df['GMM_Cluster'] = cluster_assignments
    
    # 7. Create visualizations
    cluster_means_df = visualize_results(independent_variables, cluster_assignments, optimal_k)
    
    # 8. Display summary results
    print("\n📋 ANALYSIS SUMMARY")
    print("=" * 30)
    print(f"✓ Data points analyzed: {len(df)}")
    print(f"✓ Features used: {independent_variables.shape[1]}")
    print(f"✓ Optimal clusters: {optimal_k}")
    print(f"✓ Cluster assignments added to dataframe as 'GMM_Cluster' column")
    
    print("\n🎯 Cluster Characteristics:")
    print(cluster_means_df.round(3))
    
    # Note about Mixture of Regressions
    print("\n" + "="*60)
    print("📝 NOTE ON MIXTURE OF REGRESSIONS:")
    print("A full Mixture of Regressions model was not implemented due to its complexity.")
    print("This analysis uses Gaussian Mixture Model clustering on the perception variables.")
    print("For true MoR, you would need to implement EM algorithm with:")
    print("- E-step: Calculate component responsibilities using regression likelihoods")
    print("- M-step: Update regression coefficients and variances using weighted least squares")
    print("="*60)
    
    return df, gm_optimal, cluster_means_df

if __name__ == "__main__":
    # Run the analysis
    try:
        df_result, model, cluster_summary = main()
        print("\n✅ Analysis completed successfully!")
        
        # Optionally save results
        if 'df_result' in locals():
            df_result.to_csv('mcdonalds_with_clusters.csv', index=False)
            print("💾 Results saved to 'mcdonalds_with_clusters.csv'")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("Please check that all required packages are installed:")
        print("pip install pandas numpy matplotlib seaborn scikit-learn statsmodels")