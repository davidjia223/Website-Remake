import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Optional
import joblib

class DataAnalyzer:
    """
    A class for data analysis and machine learning operations.
    Includes data preprocessing, visualization, and model training capabilities.
    """
    
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            file_path (str): Path to the data file
            file_type (str): Type of file ('csv', 'excel', 'json')
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            if file_type.lower() == 'csv':
                self.data = pd.read_csv(file_path)
            elif file_type.lower() == 'excel':
                self.data = pd.read_excel(file_path)
            elif file_type.lower() == 'json':
                self.data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, 
                       target_column: str,
                       features: List[str],
                       handle_missing: bool = True) -> tuple:
        """
        Preprocess the data for analysis
        
        Args:
            target_column (str): Name of the target variable
            features (List[str]): List of feature columns to use
            handle_missing (bool): Whether to handle missing values
            
        Returns:
            tuple: Preprocessed features and target variables
        """
        try:
            if self.data is None:
                raise ValueError("No data loaded. Please load data first.")
            
            # Handle missing values if requested
            if handle_missing:
                self.data = self.handle_missing_values(self.data)
            
            # Separate features and target
            X = self.data[features]
            y = self.data[target_column]
            
            # Scale the features
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y
            
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            return None, None
    
    @staticmethod
    def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            data (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        # Fill numeric columns with median
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
        # Fill categorical columns with mode
        categorical_columns = data.select_dtypes(exclude=[np.number]).columns
        data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
        
        return data
    
    def visualize_data(self, 
                      plot_type: str,
                      x: Optional[str] = None,
                      y: Optional[str] = None,
                      title: str = "Data Visualization") -> None:
        """
        Create various types of plots for data visualization
        
        Args:
            plot_type (str): Type of plot ('histogram', 'scatter', 'boxplot', 'correlation')
            x (str): Column name for x-axis
            y (str): Column name for y-axis
            title (str): Title of the plot
        """
        plt.figure(figsize=(10, 6))
        
        try:
            if plot_type == 'histogram':
                sns.histplot(data=self.data, x=x)
            elif plot_type == 'scatter':
                sns.scatterplot(data=self.data, x=x, y=y)
            elif plot_type == 'boxplot':
                sns.boxplot(data=self.data, x=x, y=y)
            elif plot_type == 'correlation':
                sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            plt.title(title)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
    
    def train_model(self, 
                   model,
                   X: np.ndarray,
                   y: np.ndarray,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict:
        """
        Train a machine learning model
        
        Args:
            model: Scikit-learn model object
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target variable
            test_size (float): Proportion of test set
            random_state (int): Random state for reproducibility
            
        Returns:
            Dict: Dictionary containing model performance metrics
        """
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train the model
            self.model = model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained model to a file
        
        Args:
            file_path (str): Path to save the model
        """
        try:
            if self.model is None:
                raise ValueError("No model to save. Please train a model first.")
            
            joblib.dump(self.model, file_path)
            print(f"Model saved successfully to {file_path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def load_model(self, file_path: str) -> None:
        """
        Load a saved model from a file
        
        Args:
            file_path (str): Path to the saved model
        """
        try:
            self.model = joblib.load(file_path)
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = DataAnalyzer()
    
    # Example: Load and analyze sample data
    # Replace 'sample_data.csv' with your actual data file
    try:
        # Load data
        data = analyzer.load_data('sample_data.csv')
        
        if data is not None:
            # Visualize correlations
            analyzer.visualize_data(plot_type='correlation', title='Feature Correlations')
            
            # Preprocess data
            features = ['feature1', 'feature2', 'feature3']  # Replace with your feature columns
            X, y = analyzer.preprocess_data(target_column='target', features=features)
            
            if X is not None and y is not None:
                # Train a model (example with Random Forest)
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                metrics = analyzer.train_model(model, X, y)
                if metrics:
                    print("\nModel Performance:")
                    print(f"Accuracy: {metrics['accuracy']:.4f}")
                    print("\nClassification Report:")
                    print(metrics['classification_report'])
                    
                    # Save the model
                    analyzer.save_model('trained_model.joblib')
    
    except Exception as e:
        print(f"Error in example usage: {str(e)}") 