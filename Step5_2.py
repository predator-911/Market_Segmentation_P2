# K-Means Clustering Analysis - Local/Codespace Version
# Requirements: pandas, numpy, matplotlib, seaborn, scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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
    Preprocess the McDonald's data for K-means clustering
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
    
    # Standardize the data
    print("\n📏 Standardizing data...")
    scaler = StandardScaler()
    MD_scaled = scaler.fit_transform(MD_x_encoded)
    
    print(f"✅ Preprocessing complete!")
    print(f"   - Encoded data shape: {MD_x_encoded.shape}")
    print(f"   - Scaled data shape: {MD_scaled.shape}")
    
    return MD_x_encoded, MD_scaled, scaler

def perform_elbow_analysis(MD_scaled, k_range=None):
    """
    Perform elbow method analysis for optimal k selection
    """
    if MD_scaled is None:
        return None, None
    
    if k_range is None:
        k_range = range(2, 9)
    
    print(f"\n📈 Performing Elbow Method analysis for k = {min(k_range)} to {max(k_range)}...")
    
    inertia = []
    silhouette_scores = []
    
    for k in k_range:
        print(f"   Testing k = {k}...", end=" ")
        
        # Fit K-means
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
        kmeans.fit(MD_scaled)
        
        # Store inertia
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette score
        if k > 1:  # Silhouette score requires at least 2 clusters
            sil_score = silhouette_score(MD_scaled, kmeans.labels_)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(0)
        
        print(f"Inertia: {kmeans.inertia_:.2f}, Silhouette: {silhouette_scores[-1]:.3f}")
    
    return list(k_range), inertia, silhouette_scores

def create_comprehensive_plots(k_range, inertia, silhouette_scores, save_plots=True):
    """
    Create comprehensive visualization plots
    """
    print("\n📊 Creating visualization plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('K-Means Clustering Analysis for McDonald\'s Customer Segmentation', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Scree Plot (Elbow Method)
    axes[0, 0].plot(k_range, inertia, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia (Within-cluster sum of squares)')
    axes[0, 0].set_title('Scree Plot - Elbow Method')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(k_range)
    
    # Add annotations for key points
    for i, (k, iner) in enumerate(zip(k_range, inertia)):
        axes[0, 0].annotate(f'{iner:.0f}', (k, iner), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: Silhouette Scores
    axes[0, 1].plot(k_range, silhouette_scores, marker='s', linewidth=2, markersize=8, color='#A23B72')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Score Analysis')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(k_range)
    
    # Add annotations for silhouette scores
    for i, (k, sil) in enumerate(zip(k_range, silhouette_scores)):
        axes[0, 1].annotate(f'{sil:.3f}', (k, sil), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 3: Inertia Rate of Change (to help identify elbow)
    inertia_diff = [0] + [inertia[i-1] - inertia[i] for i in range(1, len(inertia))]
    axes[1, 0].bar(k_range, inertia_diff, color='#F18F01', alpha=0.7)
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Inertia Reduction')
    axes[1, 0].set_title('Inertia Reduction per Additional Cluster')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_xticks(k_range)
    
    # Plot 4: Combined Analysis
    ax4_twin = axes[1, 1].twinx()
    
    line1 = axes[1, 1].plot(k_range, inertia, marker='o', color='#2E86AB', 
                           linewidth=2, label='Inertia')
    line2 = ax4_twin.plot(k_range, silhouette_scores, marker='s', color='#A23B72', 
                         linewidth=2, label='Silhouette Score')
    
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Inertia', color='#2E86AB')
    ax4_twin.set_ylabel('Silhouette Score', color='#A23B72')
    axes[1, 1].set_title('Combined Analysis: Inertia vs Silhouette Score')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(k_range)
    
    # Add legend
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    
    if save_plots:
        filename = 'kmeans_elbow_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Plots saved as '{filename}'")
    
    # Display plot if possible
    try:
        plt.show()
    except:
        print("📊 Plots created but display not available in this environment")
    
    return fig

def recommend_optimal_k(k_range, inertia, silhouette_scores):
    """
    Provide recommendations for optimal k based on multiple criteria
    """
    print("\n🎯 OPTIMAL K RECOMMENDATIONS:")
    print("=" * 40)
    
    # Find elbow using rate of change
    inertia_diff = [inertia[i-1] - inertia[i] for i in range(1, len(inertia))]
    inertia_diff2 = [inertia_diff[i-1] - inertia_diff[i] for i in range(1, len(inertia_diff))]
    
    # Best silhouette score
    best_sil_idx = np.argmax(silhouette_scores)
    best_sil_k = k_range[best_sil_idx]
    
    print(f"📊 Silhouette Score Analysis:")
    print(f"   Best k = {best_sil_k} (Silhouette Score: {silhouette_scores[best_sil_idx]:.3f})")
    
    # Show top 3 by silhouette score
    sil_sorted = sorted(zip(k_range, silhouette_scores), key=lambda x: x[1], reverse=True)
    print(f"   Top 3 by Silhouette Score:")
    for i, (k, score) in enumerate(sil_sorted[:3]):
        print(f"      {i+1}. k={k}: {score:.3f}")
    
    # Elbow method suggestion
    print(f"\n📈 Elbow Method Analysis:")
    print(f"   Inertia values: {[f'{x:.0f}' for x in inertia]}")
    
    # Look for the point where inertia reduction slows significantly
    if len(inertia_diff) >= 2:
        max_diff_idx = np.argmax(inertia_diff)
        elbow_k = k_range[max_diff_idx + 1]  # +1 because diff is offset
        print(f"   Suggested elbow at k = {elbow_k}")
    
    # Overall recommendation
    print(f"\n💡 OVERALL RECOMMENDATION:")
    if best_sil_k <= max(k_range) - 1:  # Not the highest k tested
        print(f"   Consider k = {best_sil_k} (highest silhouette score)")
    else:
        print(f"   Consider k = {sil_sorted[1][0]} or k = {sil_sorted[2][0]} (good silhouette scores)")
    
    print(f"   Also evaluate k = 3-5 for business interpretability")
    
    return best_sil_k

def analyze_cluster_characteristics(df, MD_x_encoded, optimal_k, MD_scaled):
    """
    Fit the optimal model and analyze cluster characteristics
    """
    print(f"\n🔍 Analyzing cluster characteristics for k = {optimal_k}...")
    
    # Fit the optimal model
    kmeans_optimal = KMeans(n_clusters=optimal_k, n_init=10, random_state=1234)
    kmeans_optimal.fit(MD_scaled)
    
    # Add cluster labels to dataframe
    df['KMeans_Cluster'] = kmeans_optimal.labels_
    
    # Calculate cluster statistics
    print(f"📊 Cluster distribution:")
    cluster_counts = pd.Series(kmeans_optimal.labels_).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   Cluster {cluster}: {count} customers ({percentage:.1f}%)")
    
    # Analyze cluster characteristics
    print(f"\n🎯 Cluster characteristics (average response rates):")
    cluster_means = []
    for i in range(optimal_k):
        mask = kmeans_optimal.labels_ == i
        cluster_mean = MD_x_encoded[mask].mean()
        cluster_means.append(cluster_mean)
        print(f"\n   Cluster {i} ({sum(mask)} customers):")
        for col, avg in cluster_mean.items():
            print(f"      {col}: {avg:.2f} ({avg*100:.0f}% Yes)")
    
    return kmeans_optimal, df

def main():
    """
    Main analysis function
    """
    print("🍟 K-Means Clustering Analysis for McDonald's Customer Segmentation")
    print("=" * 70)
    
    # Load and preprocess data
    df = load_data("mcdonalds.csv")
    MD_x_encoded, MD_scaled, scaler = preprocess_data(df)
    
    if MD_scaled is None:
        print("❌ Cannot proceed without data. Please check your dataset.")
        return None
    
    # Perform elbow analysis
    k_range, inertia, silhouette_scores = perform_elbow_analysis(MD_scaled)
    
    if k_range is None:
        print("❌ Elbow analysis failed.")
        return None
    
    # Create visualizations
    fig = create_comprehensive_plots(k_range, inertia, silhouette_scores)
    
    # Recommend optimal k
    optimal_k = recommend_optimal_k(k_range, inertia, silhouette_scores)
    
    # Analyze cluster characteristics
    kmeans_model, df_with_clusters = analyze_cluster_characteristics(
        df, MD_x_encoded, optimal_k, MD_scaled)
    
    # Save results
    output_file = 'mcdonalds_kmeans_results.csv'
    df_with_clusters.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to '{output_file}'")
    
    print("\n✅ Analysis completed successfully!")
    
    return {
        'dataframe': df_with_clusters,
        'model': kmeans_model,
        'scaler': scaler,
        'k_range': k_range,
        'inertia': inertia,
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k
    }

if __name__ == "__main__":
    try:
        results = main()
        if results:
            print(f"\n🎯 Final Results Summary:")
            print(f"   - Optimal clusters: {results['optimal_k']}")
            print(f"   - Model type: K-Means")
            print(f"   - Data points: {len(results['dataframe'])}")
            print(f"   - Features used: {results['dataframe'].iloc[:, :11].shape[1]}")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure all required packages are installed:")
        print("   pip install pandas numpy matplotlib seaborn scikit-learn")
        print("2. Make sure the McDonald's dataset is in the correct location")
        print("3. Check that the dataset has the expected format (11 binary columns)")