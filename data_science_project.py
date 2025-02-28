from advanced_data_analysis import AdvancedDataAnalyzer
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Load Boston Housing dataset as an example
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    data['PRICE'] = boston.target
    
    # Save to CSV for our analyzer to use
    data.to_csv('boston_housing.csv', index=False)
    
    # Initialize our analyzer
    analyzer = AdvancedDataAnalyzer()
    
    print("=== Starting Boston Housing Price Analysis ===\n")
    
    # 1. Load and analyze the data
    analyzer.load_and_analyze('boston_housing.csv')
    
    # 2. Perform EDA
    print("\n=== Performing Exploratory Data Analysis ===\n")
    analyzer.perform_eda(target_column='PRICE')
    
    # 3. Feature Engineering
    print("\n=== Performing Feature Engineering ===\n")
    engineered_data = analyzer.feature_engineering(
        target_column='PRICE',
        transform_method='yeo-johnson'
    )
    
    # 4. Dimensionality Reduction
    print("\n=== Performing Dimensionality Reduction ===\n")
    reduced_data = analyzer.dimensionality_reduction(
        n_components=2,
        plot=True
    )
    
    # 5. Model Training
    print("\n=== Training Random Forest Model ===\n")
    X = engineered_data.drop('PRICE', axis=1)
    y = engineered_data['PRICE']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 6. Feature Importance Analysis
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # 7. Model Evaluation
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    print("\nAnalysis complete! Check the visualizations and statistics above.")

if __name__ == "__main__":
    main() 