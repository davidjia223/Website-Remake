import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from typing import List, Dict, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataAnalyzer:
    """
    Advanced data analysis toolkit for exploratory data analysis (EDA),
    feature engineering, and statistical analysis.
    """
    
    def __init__(self):
        self.data = None
        self.original_data = None
        self.numeric_columns = None
        self.categorical_columns = None
        
    def load_and_analyze(self, file_path: str) -> None:
        """
        Load data and perform initial analysis
        
        Args:
            file_path (str): Path to the data file
        """
        try:
            # Load data
            self.data = pd.read_csv(file_path)
            self.original_data = self.data.copy()
            
            # Identify column types
            self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            self.categorical_columns = self.data.select_dtypes(exclude=[np.number]).columns
            
            # Print initial analysis
            self._print_initial_analysis()
            
        except Exception as e:
            print(f"Error in load_and_analyze: {str(e)}")
    
    def _print_initial_analysis(self) -> None:
        """Print initial data analysis results"""
        print("\n=== Initial Data Analysis ===")
        print(f"\nDataset Shape: {self.data.shape}")
        print("\nColumn Types:")
        print(self.data.dtypes)
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        print("\nBasic Statistics:")
        print(self.data.describe())
    
    def perform_eda(self, target_column: Optional[str] = None) -> None:
        """
        Perform comprehensive exploratory data analysis
        
        Args:
            target_column (str): Name of the target variable for analysis
        """
        try:
            # Create a figure for multiple plots
            plt.figure(figsize=(15, 10))
            
            # 1. Distribution Analysis
            self._analyze_distributions()
            
            # 2. Correlation Analysis
            self._analyze_correlations()
            
            # 3. Target Analysis (if target provided)
            if target_column:
                self._analyze_target_relationships(target_column)
            
            # 4. Outlier Analysis
            self._analyze_outliers()
            
            # 5. Category Analysis
            self._analyze_categories()
            
        except Exception as e:
            print(f"Error in perform_eda: {str(e)}")
    
    def _analyze_distributions(self) -> None:
        """Analyze and plot distributions of numeric variables"""
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(self.numeric_columns[:5], 1):
            plt.subplot(1, 5, i)
            sns.histplot(self.data[col], kde=True)
            plt.title(f'{col} Distribution')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Skewness and Kurtosis
        print("\n=== Distribution Statistics ===")
        for col in self.numeric_columns:
            skew = stats.skew(self.data[col].dropna())
            kurt = stats.kurtosis(self.data[col].dropna())
            print(f"\n{col}:")
            print(f"Skewness: {skew:.2f}")
            print(f"Kurtosis: {kurt:.2f}")
    
    def _analyze_correlations(self) -> None:
        """Analyze and plot correlations between variables"""
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.data[self.numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Print high correlations
        high_corr = self._get_high_correlations(correlation_matrix, threshold=0.7)
        if high_corr:
            print("\n=== High Correlations ===")
            for pair, corr in high_corr.items():
                print(f"{pair}: {corr:.2f}")
    
    def _analyze_target_relationships(self, target_column: str) -> None:
        """
        Analyze relationships between features and target variable
        
        Args:
            target_column (str): Name of the target variable
        """
        plt.figure(figsize=(15, 5))
        
        # For numeric features
        for i, col in enumerate(self.numeric_columns[:3], 1):
            if col != target_column:
                plt.subplot(1, 3, i)
                sns.scatterplot(data=self.data, x=col, y=target_column)
                plt.title(f'{col} vs {target_column}')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # For categorical features
        if len(self.categorical_columns) > 0:
            plt.figure(figsize=(15, 5))
            for i, col in enumerate(self.categorical_columns[:3], 1):
                plt.subplot(1, 3, i)
                sns.boxplot(data=self.data, x=col, y=target_column)
                plt.title(f'{target_column} by {col}')
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    def _analyze_outliers(self) -> Dict[str, List[float]]:
        """
        Analyze outliers in numeric variables
        
        Returns:
            Dict[str, List[float]]: Dictionary with outlier information
        """
        outliers_info = {}
        
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(self.numeric_columns[:5], 1):
            plt.subplot(1, 5, i)
            sns.boxplot(y=self.data[col])
            plt.title(f'{col} Boxplot')
        plt.tight_layout()
        plt.show()
        
        print("\n=== Outlier Analysis ===")
        for col in self.numeric_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)][col]
            
            outliers_info[col] = [lower_bound, upper_bound]
            print(f"\n{col}:")
            print(f"Number of outliers: {len(outliers)}")
            print(f"Outlier boundaries: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return outliers_info
    
    def _analyze_categories(self) -> None:
        """Analyze categorical variables"""
        if len(self.categorical_columns) > 0:
            plt.figure(figsize=(15, 5))
            for i, col in enumerate(self.categorical_columns[:3], 1):
                plt.subplot(1, 3, i)
                value_counts = self.data[col].value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f'{col} Distribution')
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            print("\n=== Category Analysis ===")
            for col in self.categorical_columns:
                print(f"\n{col}:")
                print(self.data[col].value_counts())
    
    @staticmethod
    def _get_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> Dict[tuple, float]:
        """
        Get pairs of highly correlated features
        
        Args:
            corr_matrix (pd.DataFrame): Correlation matrix
            threshold (float): Correlation threshold
            
        Returns:
            Dict[tuple, float]: Dictionary of feature pairs and their correlations
        """
        high_corr = {}
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr[(corr_matrix.index[i], corr_matrix.columns[j])] = corr_matrix.iloc[i, j]
        return high_corr
    
    def feature_engineering(self, 
                          target_column: Optional[str] = None,
                          transform_method: str = 'yeo-johnson') -> pd.DataFrame:
        """
        Perform feature engineering
        
        Args:
            target_column (str): Name of target variable
            transform_method (str): Method for power transform ('yeo-johnson' or 'box-cox')
            
        Returns:
            pd.DataFrame: Engineered features
        """
        try:
            engineered_data = self.data.copy()
            
            # 1. Handle missing values
            engineered_data = self._handle_missing_values(engineered_data)
            
            # 2. Transform skewed features
            engineered_data = self._transform_skewed_features(engineered_data, transform_method)
            
            # 3. Create interaction features
            engineered_data = self._create_interactions(engineered_data)
            
            # 4. Encode categorical variables
            engineered_data = self._encode_categories(engineered_data)
            
            # 5. Feature selection (if target provided)
            if target_column:
                engineered_data = self._select_features(engineered_data, target_column)
            
            return engineered_data
            
        except Exception as e:
            print(f"Error in feature_engineering: {str(e)}")
            return self.data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Numeric columns: fill with median
        for col in self.numeric_columns:
            data[col] = data[col].fillna(data[col].median())
        
        # Categorical columns: fill with mode
        for col in self.categorical_columns:
            data[col] = data[col].fillna(data[col].mode()[0])
        
        return data
    
    def _transform_skewed_features(self, 
                                 data: pd.DataFrame,
                                 method: str = 'yeo-johnson') -> pd.DataFrame:
        """Transform skewed numeric features"""
        pt = PowerTransformer(method=method)
        
        for col in self.numeric_columns:
            if abs(stats.skew(data[col].dropna())) > 0.5:
                data[f"{col}_transformed"] = pt.fit_transform(data[[col]])
        
        return data
    
    def _create_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numeric variables"""
        numeric_cols = list(self.numeric_columns)[:5]  # Limit to prevent explosion of features
        
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                data[f"{col1}_{col2}_interaction"] = data[col1] * data[col2]
        
        return data
    
    def _encode_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        return pd.get_dummies(data, columns=self.categorical_columns)
    
    def _select_features(self, 
                        data: pd.DataFrame,
                        target_column: str,
                        k: int = 10) -> pd.DataFrame:
        """Select top k features based on ANOVA F-value"""
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        return data[[target_column] + selected_features]
    
    def dimensionality_reduction(self, 
                               n_components: int = 2,
                               plot: bool = True) -> np.ndarray:
        """
        Perform PCA for dimensionality reduction
        
        Args:
            n_components (int): Number of components
            plot (bool): Whether to plot the results
            
        Returns:
            np.ndarray: Transformed data
        """
        try:
            # Prepare data
            X = self.data[self.numeric_columns]
            X = StandardScaler().fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            
            # Plot results
            if plot:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                        np.cumsum(pca.explained_variance_ratio_))
                plt.xlabel('Number of Components')
                plt.ylabel('Cumulative Explained Variance Ratio')
                plt.title('PCA Analysis')
                plt.show()
                
                print("\n=== PCA Analysis ===")
                print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
                print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
            
            return X_pca
            
        except Exception as e:
            print(f"Error in dimensionality_reduction: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AdvancedDataAnalyzer()
    
    try:
        # Load and analyze data
        # Replace with your dataset
        analyzer.load_and_analyze('your_dataset.csv')
        
        # Perform EDA
        analyzer.perform_eda(target_column='target_variable')
        
        # Perform feature engineering
        engineered_data = analyzer.feature_engineering(target_column='target_variable')
        
        # Perform dimensionality reduction
        reduced_data = analyzer.dimensionality_reduction(n_components=2)
        
        print("\nAnalysis complete! Check the visualizations and printed statistics above.")
        
    except Exception as e:
        print(f"Error in example usage: {str(e)}") 