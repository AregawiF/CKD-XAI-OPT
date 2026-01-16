import numpy as np
import pandas as pd
import shap

def fitness_function(x, model, X_instance, X_train, full_shap, full_pred, 
                     explainer=None, alpha=1.0, beta=1.0, gamma=0.1):
    """
    Compute fitness for a feature subset.
    
    Parameters:
    -----------
    x : array-like
        Binary vector representing selected features (1 = selected, 0 = not selected)
    model : sklearn model
        Trained model
    X_instance : pandas DataFrame
        Single instance to explain (one row)
    X_train : pandas DataFrame
        Training data for computing baseline (mean values)
    full_shap : numpy array
        Full SHAP values for the instance (positive class)
    full_pred : float
        Full model prediction probability for positive class
    explainer : shap.TreeExplainer, optional
        Pre-computed explainer (for efficiency)
    alpha : float
        Weight for prediction fidelity term
    beta : float
        Weight for SHAP fidelity term
    gamma : float
        Weight for sparsity term
    
    Returns:
    --------
    float : Fitness value (lower is better)
    """
    # Ensure at least one feature is selected
    if np.sum(x) == 0:
        return 1e10  # Very high penalty
    
    # Convert to boolean mask
    mask = np.array(x, dtype=bool)
    
    # Ensure mask length matches number of features
    if len(mask) != X_instance.shape[1]:
        raise ValueError(f"Mask length {len(mask)} doesn't match number of features {X_instance.shape[1]}")
    
    # Compute baseline (mean from training data)
    baseline = X_train.mean(axis=0)
    
    # Create masked instance: replace unselected features with baseline
    X_masked = X_instance.copy()
    # Get unselected feature names
    unselected_cols = X_instance.columns[~mask]
    for col in unselected_cols:
        # Convert baseline value to float to avoid dtype warnings
        baseline_val = float(baseline[col])
        # Convert column to float if needed to avoid dtype incompatibility
        if X_masked[col].dtype != float:
            X_masked[col] = X_masked[col].astype(float)
        X_masked.loc[:, col] = baseline_val
    
    # Prediction fidelity: difference between full and masked prediction
    y_masked = model.predict_proba(X_masked)[0, 1]  # Positive class probability
    pred_error = np.abs(full_pred - y_masked)
    
    # SHAP reconstruction error: difference between full and masked SHAP values
    # Reuse the same SHAP explainer (do not reinitialize)
    if explainer is None:
        explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for masked instance using the same explainer
    shap_values_masked = explainer.shap_values(X_masked)
    
    # Extract positive class (class 1) and ensure 1D shape (n_features,)
    # Handle both list and array formats
    if isinstance(shap_values_masked, list):
        shap_masked = np.array(shap_values_masked[1])  # Positive class
    else:
        shap_masked = np.array(shap_values_masked)
    
    # Handle different shapes to extract positive class as 1D vector
    if len(shap_masked.shape) == 1:
        # Already 1D
        pass
    elif len(shap_masked.shape) == 2:
        if shap_masked.shape[1] == 2:
            # Shape (n_features, 2) -> take positive class column (index 1)
            shap_masked = shap_masked[:, 1]
        elif shap_masked.shape[0] == 1:
            # Shape (1, n_features) -> take first row
            shap_masked = shap_masked[0]
        else:
            shap_masked = shap_masked.flatten()
    elif len(shap_masked.shape) == 3:
        # Shape (1, n_features, 2) -> take [0, :, 1] for positive class
        if shap_masked.shape[2] == 2:
            shap_masked = shap_masked[0, :, 1]
        else:
            shap_masked = shap_masked.flatten()
    else:
        shap_masked = shap_masked.flatten()
    
    # Final flatten to ensure 1D
    shap_masked = np.array(shap_masked).flatten()
    
    # Ensure full_shap is also 1D
    full_shap_1d = np.array(full_shap).flatten()
    
    # Ensure same length
    if len(full_shap_1d) != len(shap_masked):
        raise ValueError(f"SHAP value length mismatch: full_shap={len(full_shap_1d)}, masked={len(shap_masked)}")
    
    # Compute normalized SHAP error
    shap_error = np.linalg.norm(full_shap_1d - shap_masked) / (np.linalg.norm(full_shap_1d) + 1e-8)
    
    # Sparsity penalty: ratio of selected features
    sparsity = np.sum(x) / len(x)
    
    # Combined fitness (lower is better)
    fitness = alpha * pred_error + beta * shap_error + gamma * sparsity
    
    return fitness
