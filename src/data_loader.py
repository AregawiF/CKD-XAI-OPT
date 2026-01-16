import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

def load_data():
    """
    Load UCI Heart Disease dataset using ucimlrepo.
    Returns training and test sets with proper preprocessing.
    """
    # Fetch dataset
    heart_disease = fetch_ucirepo(id=45)
    
    # Extract features and targets
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Handle missing values - use mean imputation for numeric columns
    print(f"Missing values before handling:\n{X.isnull().sum()}")
    X = X.fillna(X.mean(numeric_only=True))
    
    # For any remaining object columns, fill with mode
    for col in X.select_dtypes(include=['object']).columns:
        mode_val = X[col].mode()
        if len(mode_val) > 0:
            X[col] = X[col].fillna(mode_val[0])
        else:
            X[col] = X[col].fillna(0)
    
    # Ensure all features are numeric
    X = pd.get_dummies(X, drop_first=True) if X.select_dtypes(include=['object']).shape[1] > 0 else X
    
    # Binarize target: 0 = no disease, 1-4 = disease
    y_values = y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()
    y_binary = (y_values > 0).astype(int)
    
    print(f"\nTarget distribution: {np.bincount(y_binary)}")
    print(f"Features shape: {X.shape}")
    print(f"All features numeric: {X.select_dtypes(exclude=[np.number]).shape[1] == 0}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, heart_disease
