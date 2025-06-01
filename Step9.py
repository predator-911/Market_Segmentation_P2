import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

def load_data(file_path):
    """
    Load data from CSV file.
    Replace this function with your actual data loading logic.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the path.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def fit_gmm_clusters(df, n_components=7, feature_columns=None):
    """
    Fit GMM clusters if not already present in the dataframe.
    
    Parameters:
    df: DataFrame containing the data
    n_components: Number of GMM components
    feature_columns: List of column names to use for clustering (first 11 columns by default)
    """
    if 'GMM_Cluster' not in df.columns:
        print("GMM_Cluster column not found. Fitting GMM model...")
        
        # Use first 11 columns by default or specified columns
        if feature_columns is None:
            feature_columns = df.columns[:11]
        
        # Convert categorical to numerical (assuming Yes/No encoding)
        MD_x = df[feature_columns].copy()
        for col in MD_x.columns:
            if MD_x[col].dtype == 'object':
                MD_x[col] = MD_x[col].map({'Yes': 1, 'No': 0})
        
        # Fit GMM
        gm_optimal = GaussianMixture(n_components=n_components, random_state=1234, n_init=10)
        gm_optimal.fit(MD_x)
        df['GMM_Cluster'] = gm_optimal.predict(MD_x)
        print(f"GMM clustering completed with {n_components} components.")
    else:
        print("GMM_Cluster column found in data.")
    
    return df

def analyze_perception_scores(df, independent_variables):
    """
    Analyze and visualize mean perception scores per GMM cluster.
    """
    print("\nMean of perception variables within each GMM cluster (using 0/1 encoded data):")
    
    # Create a temporary DataFrame combining the cluster labels and the 0/1 encoded perception variables
    df_encoded_with_clusters = independent_variables.copy()
    df_encoded_with_clusters['GMM_Cluster'] = df['GMM_Cluster']
    
    # Group by cluster and calculate the mean for perception columns
    cluster_perception_means = df_encoded_with_clusters.groupby('GMM_Cluster').mean()
    print(cluster_perception_means)
    
    # Visualize the mean perception scores per cluster
    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_perception_means.T, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title('Mean Perception Scores per GMM Cluster')
    plt.xlabel('GMM Cluster')
    plt.ylabel('Perception Variable')
    plt.tight_layout()
    plt.show()
    
    return cluster_perception_means

def create_segment_evaluation_plot(df):
    """
    Create segment evaluation plot based on GMM clusters.
    """
    # Create a numerical mapping for VisitFrequency
    visit_frequency_mapping = {
        'Never': 0,
        'Once a year': 1,
        'Every three months': 2,
        'Once a month': 3,
        'Once a week': 4,
        'More than once a week': 5
    }
    
    df['VisitFrequency_numeric'] = df['VisitFrequency'].map(visit_frequency_mapping)
    
    # Calculate the mean VisitFrequency_numeric for each GMM cluster
    cluster_visit_frequency_mean = df.groupby('GMM_Cluster')['VisitFrequency_numeric'].mean()
    
    # Create Like_numeric if not already present
    if 'Like_numeric' not in df.columns:
        like_mapping = {
            'I hate it!-5': -5, '-4': -4, '-3': -3, '-2': -2, '-1': -1,
            '0': 0, '+1': 1, '+2': 2, '+3': 3, '+4': 4, 'I love it!+5': 5
        }
        df['Like_numeric'] = df['Like'].map(like_mapping)
    
    # Calculate the mean Like_numeric for each GMM cluster
    cluster_like_numeric_mean = df.groupby('GMM_Cluster')['Like_numeric'].mean()
    
    # Calculate the proportion of Female in each GMM cluster
    cluster_gender_proportion = df.groupby('GMM_Cluster')['Gender'].value_counts(normalize=True).unstack(fill_value=0)['Female']
    
    # Combine the calculated values into a DataFrame for easier plotting
    segment_evaluation_data = pd.DataFrame({
        'Mean_VisitFrequency': cluster_visit_frequency_mean,
        'Mean_Like': cluster_like_numeric_mean,
        'Proportion_Female': cluster_gender_proportion
    })
    
    print("Segment Evaluation Data:")
    print(segment_evaluation_data)
    
    # Create the plot
    plt.figure(figsize=(10, 7))
    
    # The size of the bubble will represent the Proportion_Female, scale it for better visualization
    bubble_size = segment_evaluation_data['Proportion_Female'] * 2000  # Adjust scaling as needed
    
    plt.scatter(segment_evaluation_data['Mean_VisitFrequency'], 
               segment_evaluation_data['Mean_Like'],
               s=bubble_size, alpha=0.7, edgecolors="w", linewidth=1)
    
    # Add labels for each cluster
    for i, cluster in segment_evaluation_data.iterrows():
        plt.text(cluster['Mean_VisitFrequency'], cluster['Mean_Like'], 
                str(i), ha='center', va='center', fontweight='bold')
    
    plt.xlabel('Mean Visit Frequency (Numeric)')
    plt.ylabel('Mean Like (Numeric)')
    plt.title('Segment Evaluation Plot (GMM Clusters)')
    plt.grid(True, alpha=0.3)
    
    # Add legend for bubble size
    plt.figtext(0.02, 0.02, 'Bubble size represents proportion of females in cluster', 
                fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()
    
    return segment_evaluation_data

def customize_marketing_mix(df, cluster_perception_means, segment_evaluation_data, target_cluster=None):
    """
    Step 9: Customize Marketing Mix based on cluster analysis.
    Analyze clusters and provide marketing mix recommendations.
    """
    print("\n" + "="*60)
    print("STEP 9: CUSTOMIZING THE MARKETING MIX")
    print("="*60)
    
    # Analyze each cluster
    cluster_profiles = {}
    
    for cluster_id in df['GMM_Cluster'].unique():
        cluster_data = segment_evaluation_data.loc[cluster_id]
        
        # Get cluster characteristics
        visit_freq = cluster_data['Mean_VisitFrequency']
        like_score = cluster_data['Mean_Like']
        female_prop = cluster_data['Proportion_Female']
        
        # Get perception scores for this cluster
        if cluster_id in cluster_perception_means.index:
            perception_scores = cluster_perception_means.loc[cluster_id]
            
            # Identify key characteristics based on perception scores
            high_perceptions = perception_scores[perception_scores > 0.7].index.tolist()
            low_perceptions = perception_scores[perception_scores < 0.3].index.tolist()
        else:
            high_perceptions = []
            low_perceptions = []
        
        cluster_profiles[cluster_id] = {
            'visit_frequency': visit_freq,
            'like_score': like_score,
            'female_proportion': female_prop,
            'high_perceptions': high_perceptions,
            'low_perceptions': low_perceptions,
            'size': len(df[df['GMM_Cluster'] == cluster_id])
        }
    
    # Display cluster profiles
    print("\nCLUSTER PROFILES:")
    print("-" * 40)
    
    for cluster_id, profile in cluster_profiles.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {profile['size']} customers")
        print(f"  Visit Frequency: {profile['visit_frequency']:.2f}")
        print(f"  Like Score: {profile['like_score']:.2f}")
        print(f"  Female Proportion: {profile['female_proportion']:.2f}")
        print(f"  High Perceptions: {profile['high_perceptions']}")
        print(f"  Low Perceptions: {profile['low_perceptions']}")
    
    # Recommend target cluster if not specified
    if target_cluster is None:
        # Find clusters with positive like scores and reasonable visit frequency
        viable_clusters = []
        for cluster_id, profile in cluster_profiles.items():
            if profile['like_score'] > 0 and profile['visit_frequency'] > 1:
                viable_clusters.append((cluster_id, profile['like_score'], profile['size']))
        
        if viable_clusters:
            # Sort by like score and size
            viable_clusters.sort(key=lambda x: (x[1], x[2]), reverse=True)
            target_cluster = viable_clusters[0][0]
            print(f"\nRECOMMENDED TARGET CLUSTER: {target_cluster}")
        else:
            target_cluster = max(cluster_profiles.keys(), key=lambda x: cluster_profiles[x]['size'])
            print(f"\nDEFAULT TARGET CLUSTER (largest): {target_cluster}")
    
    # Generate Marketing Mix Strategy
    print(f"\nMARKETING MIX STRATEGY FOR CLUSTER {target_cluster}:")
    print("=" * 50)
    
    target_profile = cluster_profiles[target_cluster]
    
    # Product Strategy
    print("\n1. PRODUCT STRATEGY:")
    if 'expensive' in [p.lower() for p in target_profile['low_perceptions']]:
        print("   - Develop budget-friendly product line")
        print("   - Focus on value-for-money positioning")
    if 'healthy' in [p.lower() for p in target_profile['low_perceptions']]:
        print("   - Introduce healthier menu options")
        print("   - Highlight nutritional information")
    if 'tasty' in [p.lower() for p in target_profile['high_perceptions']]:
        print("   - Emphasize taste and flavor in product development")
        print("   - Maintain current recipe standards")
    
    # Price Strategy
    print("\n2. PRICE STRATEGY:")
    if target_profile['like_score'] > 2 and target_profile['visit_frequency'] < 2:
        print("   - Premium pricing strategy (high satisfaction, low frequency)")
        print("   - Bundle deals to increase visit frequency")
    elif target_profile['like_score'] > 0 and target_profile['visit_frequency'] > 3:
        print("   - Loyalty pricing programs")
        print("   - Frequent visitor discounts")
    else:
        print("   - Competitive pricing strategy")
        print("   - Value meal promotions")
    
    # Promotion Strategy
    print("\n3. PROMOTION STRATEGY:")
    if target_profile['female_proportion'] > 0.6:
        print("   - Target female-oriented media channels")
        print("   - Social media campaigns on platforms popular with women")
    elif target_profile['female_proportion'] < 0.4:
        print("   - Target male-oriented media channels")
        print("   - Sports and gaming-related advertising")
    else:
        print("   - Gender-neutral advertising approach")
    
    if target_profile['visit_frequency'] < 2:
        print("   - Awareness and trial campaigns")
        print("   - First-time visitor incentives")
    else:
        print("   - Retention and loyalty campaigns")
        print("   - Referral programs")
    
    # Place Strategy
    print("\n4. PLACE STRATEGY:")
    if target_profile['visit_frequency'] > 3:
        print("   - Ensure convenient locations for frequent visitors")
        print("   - Express service options")
        print("   - Mobile ordering and delivery")
    else:
        print("   - Focus on high-visibility locations")
        print("   - Accessible parking and public transport")
    
    # Implementation Timeline
    print("\n5. IMPLEMENTATION TIMELINE:")
    print("   - Month 1-2: Market research and product development")
    print("   - Month 3-4: Pricing strategy implementation")
    print("   - Month 5-6: Launch promotional campaigns")
    print("   - Month 7-12: Monitor and adjust strategies")
    
    # Success Metrics
    print("\n6. SUCCESS METRICS:")
    print(f"   - Increase visit frequency for Cluster {target_cluster} by 20%")
    print(f"   - Improve like score for Cluster {target_cluster} by 15%")
    print("   - Monitor cluster migration and retention rates")
    print("   - Track revenue per customer in target cluster")
    
    return target_cluster, cluster_profiles

def main():
    """
    Main function to run the segment analysis.
    """
    # Load your data here - replace with your actual file path
    file_path = "mcdonalds.csv"  # Change this to your data file path
    
    df = load_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Display basic info about the data
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Fit GMM clusters if needed
    df = fit_gmm_clusters(df, n_components=7)  # Adjust n_components as needed
    
    # Check if required columns exist
    required_columns = ['VisitFrequency', 'Like', 'Gender', 'GMM_Cluster']
    missing_columns = [col for col in required_columns if col not in required_columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print("Please ensure your data has the required columns or modify the column names in the script.")
        return
    
    # Create independent variables (0/1 encoded perception data)
    # Assuming first 11 columns are perception variables - adjust as needed
    perception_columns = df.columns[:11]  # Modify this based on your actual perception columns
    independent_variables = df[perception_columns].copy()
    
    # Convert to 0/1 encoding if not already done
    for col in independent_variables.columns:
        if independent_variables[col].dtype == 'object':
            independent_variables[col] = independent_variables[col].map({'Yes': 1, 'No': 0})
    
    # Analyze perception scores per cluster
    cluster_perception_means = analyze_perception_scores(df, independent_variables)
    
    # Create segment evaluation plot
    segment_data = create_segment_evaluation_plot(df)
    
    # Step 9: Customize Marketing Mix
    target_cluster, cluster_profiles = customize_marketing_mix(
        df, cluster_perception_means, segment_data
    )
    
    # Save all results
    segment_data.to_csv('segment_evaluation_results.csv')
    cluster_perception_means.to_csv('cluster_perception_means.csv')
    
    # Save marketing mix analysis
    with open('marketing_mix_analysis.txt', 'w') as f:
        f.write(f"Marketing Mix Analysis Report\n")
        f.write(f"Target Cluster: {target_cluster}\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
        
        for cluster_id, profile in cluster_profiles.items():
            f.write(f"Cluster {cluster_id} Profile:\n")
            f.write(f"  Size: {profile['size']} customers\n")
            f.write(f"  Visit Frequency: {profile['visit_frequency']:.2f}\n")
            f.write(f"  Like Score: {profile['like_score']:.2f}\n")
            f.write(f"  Female Proportion: {profile['female_proportion']:.2f}\n")
            f.write(f"  High Perceptions: {profile['high_perceptions']}\n")
            f.write(f"  Low Perceptions: {profile['low_perceptions']}\n\n")
    
    print("\nResults saved:")
    print("- segment_evaluation_results1.csv")
    print("- cluster_perception_means.csv") 
    print("- marketing_mix_analysis.txt")

if __name__ == "__main__":
    main()