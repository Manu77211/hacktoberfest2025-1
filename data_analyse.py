import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(data):
    """
    Clean the data by handling missing values and duplicates.
    
    Args:
        data (pd.DataFrame): Input data.
    
    Returns:
        pd.DataFrame: Cleaned data.
    """
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Fill missing values with mean for numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    
    # Fill missing values with mode for categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    print("Data cleaned successfully.")
    return data

def analyze_data(data):
    """
    Perform basic statistical analysis on the data.
    
    Args:
        data (pd.DataFrame): Input data.
    """
    print("Basic Statistics:")
    print(data.describe())
    
    print("\nCorrelation Matrix:")
    numeric_data = data.select_dtypes(include=[np.number])
    correlation = numeric_data.corr()
    print(correlation)
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def visualize_data(data, x_col, y_col):
    """
    Create visualizations for the data.
    
    Args:
        data (pd.DataFrame): Input data.
        x_col (str): Column for x-axis.
        y_col (str): Column for y-axis.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(data[x_col], data[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    
    plt.subplot(1, 2, 2)
    data[y_col].hist(bins=30)
    plt.xlabel(y_col)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {y_col}')
    
    plt.tight_layout()
    plt.show()

def build_model(data, target_col):
    """
    Build a simple linear regression model.
    
    Args:
        data (pd.DataFrame): Input data.
        target_col (str): Target column for prediction.
    
    Returns:
        model: Trained model.
    """
    # Prepare features and target
    X = data.drop(target_col, axis=1).select_dtypes(include=[np.number])
    y = data[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    
    return model

def main():
    """
    Main function to run the data analysis pipeline.
    """
    # Example usage
    file_path = 'data.csv'  # Replace with actual file path
    data = load_data(file_path)
    
    if data is not None:
        data = clean_data(data)
        analyze_data(data)
        
        # Assuming 'price' and 'area' columns exist
        if 'area' in data.columns and 'price' in data.columns:
            visualize_data(data, 'area', 'price')
            model = build_model(data, 'price')
        else:
            print("Required columns not found for visualization and modeling.")
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()
