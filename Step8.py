import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

def load_data(file_path):
    """
    Load data from CSV file.
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
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print("Please ensure your data has the required columns or modify the column names in the script.")
        return
    
    # Create segment evaluation plot
    segment_data = create_segment_evaluation_plot(df)
    
    # Save results
    segment_data.to_csv('segment_evaluation_results.csv')
    print("Results saved to 'segment_evaluation_results.csv'")

if __name__ == "__main__":
    main()
