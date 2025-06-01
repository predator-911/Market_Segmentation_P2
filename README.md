# McDonald's Customer Segmentation Analysis

A comprehensive customer segmentation analysis of McDonald's customer data using advanced machine learning clustering techniques to identify distinct customer segments and their behavioral patterns.

## 🎯 Project Overview

This project performs an in-depth segmentation analysis on McDonald's customer data to understand different customer profiles based on their perceptions, demographics, and behavioral patterns. The analysis employs multiple clustering algorithms to identify meaningful customer segments that can inform targeted marketing strategies.

## 📊 Key Features

- **Multi-Algorithm Clustering**: Implements both K-Means and Gaussian Mixture Models (GMM)
- **Comprehensive Data Preprocessing**: Converts categorical data to numerical formats for analysis
- **Model Selection**: Uses statistical criteria (AIC/BIC) and visualization techniques (Elbow method) for optimal cluster selection
- **Segment Profiling**: Detailed analysis of each customer segment's characteristics
- **Interactive Visualizations**: Clear plots and charts for segment evaluation and comparison

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning algorithms
- **statsmodels** - Statistical modeling

## 📋 Prerequisites

Before running this project, ensure you have Python 3.8 or higher installed on your system.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mcdonalds-segmentation.git
   cd mcdonalds-segmentation
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place your data file:**
   Ensure `mcdonalds.csv` is in the project root directory.

## 🏃‍♂️ Running the Analysis

### Option 1: Run the complete analysis
```bash
python main.py
```

### Option 2: Run individual components
```bash
# Data preprocessing only
python -c "from src.data_preprocessing import preprocess_data; preprocess_data('data/mcdonalds.csv')"

# Clustering analysis only
python -c "from src.clustering_analysis import perform_clustering; perform_clustering()"
```

### Option 3: Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/mcdonalds_analysis.ipynb
```

## 📈 Analysis Workflow

### 1. Data Loading and Inspection
- Loads the McDonald's customer dataset
- Performs initial data quality assessment
- Examines data types, missing values, and unique values
- Analyzes key variables like 'Gender' and 'Like'

### 2. Data Preprocessing
- **Perception Variables**: Converts 11 perception attributes ('yummy', 'convenient', etc.) from 'Yes'/'No' to binary format (1/0)
- **Like Variable**: Transforms ordinal 'Like' ratings to numerical scale (-5 to +5)
- **Data Validation**: Ensures data quality and consistency

### 3. Clustering Analysis

#### K-Means Clustering
- Applies K-Means algorithm to perception data
- Uses Elbow method (Scree plot) for optimal cluster determination
- Evaluates within-cluster sum of squares (WCSS)

#### Gaussian Mixture Model (GMM)
- Implements GMM clustering on perception data
- Utilizes AIC and BIC information criteria for model selection
- Identifies optimal number of components (7 clusters based on BIC)

### 4. Segment Analysis
Comprehensive profiling of identified segments including:
- **Perception Profiles**: Mean scores for each of the 11 perception attributes
- **Demographics**: Age distribution and gender composition
- **Behavioral Patterns**: Visit frequency and brand affinity ('Like' ratings)
- **Statistical Summaries**: Descriptive statistics for each segment

### 5. Visualization and Evaluation
- Segment evaluation plots showing relationship between visit frequency, likeability, and demographics
- Cluster comparison visualizations
- Detailed segment characteristic charts

## 📊 Key Findings

The analysis reveals **7 distinct customer segments** with unique characteristics:

1. **Segment Differentiation**: Each cluster exhibits distinct perception patterns of McDonald's attributes
2. **Demographic Diversity**: Segments vary significantly in age profiles and gender distribution
3. **Behavioral Variations**: Different visit frequencies and brand preferences across segments
4. **Actionable Insights**: Clear profiles enable targeted marketing strategies

## 🔍 Results Interpretation

- **Perception Analysis**: Identifies which McDonald's attributes resonate with different customer groups
- **Customer Profiling**: Provides detailed demographic and behavioral profiles for each segment
- **Marketing Opportunities**: Reveals potential for segment-specific marketing campaigns
- **Business Intelligence**: Offers data-driven insights for strategic decision-making

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
statsmodels>=0.12.0
jupyter>=1.0.0
```

## 🚀 Future Enhancements

- **Advanced Modeling**: Implement Expectation-Maximization algorithm for Mixture of Linear Regressions
- **Deep Clustering**: Explore deep learning-based clustering approaches
- **Real-time Analysis**: Develop streaming analytics capabilities
- **Interactive Dashboard**: Create web-based dashboard for dynamic segment exploration
- **Predictive Modeling**: Build models to predict customer segment membership
- **A/B Testing Framework**: Implement framework for testing segment-specific strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/predator-911/Market_Segmentation_P2/issues) page
2. Create a new issue with detailed description
3. Contact: your.email@example.com

## 📚 References

- Clustering Analysis in Marketing Research
- Gaussian Mixture Models for Customer Segmentation
- K-Means Clustering Best Practices
- Customer Segmentation Strategies in Fast Food Industry

## 🏆 Acknowledgments

- Dataset source: McDonald's Customer Research
- Inspiration: Marketing Analytics and Customer Segmentation Literature
- Community: Open-source contributors and researchers

---

**Note**: This analysis is for educational and research purposes. Ensure you have proper permissions to use the dataset in commercial applications.
