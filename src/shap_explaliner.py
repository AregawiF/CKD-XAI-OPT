import shap
import numpy as np

def compute_shap(rf_model, X_instance, explainer=None):
    """
    Compute SHAP values for a given instance.
    Returns SHAP values for the positive class (class 1) as a 1D vector of shape (n_features,).
    
    Parameters:
    -----------
    rf_model : sklearn model
        Trained model
    X_instance : pandas DataFrame
        Single instance to explain (one row)
    explainer : shap.TreeExplainer, optional
        Pre-computed explainer (for efficiency). If None, creates a new one.
    
    Returns:
    --------
    shap_positive : numpy array
        1D array of SHAP values for positive class, shape (n_features,)
    explainer : shap.TreeExplainer
        The explainer object (reused or newly created)
    """
    # Reuse explainer if provided, otherwise create new one
    if explainer is None:
        explainer = shap.TreeExplainer(rf_model)
    
    shap_values = explainer.shap_values(X_instance)
    
    # For binary classification, shap_values can be:
    # 1. A list [class_0, class_1] where each is an array
    # 2. A numpy array with shape (n_features, 2) where columns are [class_0, class_1]
    # 3. A numpy array with shape (1, n_features, 2) for single instance
    
    # Convert to numpy array for easier handling
    if isinstance(shap_values, list):
        # List format: [class_0_array, class_1_array]
        shap_positive = np.array(shap_values[1])  # Positive class (class 1)
    else:
        # Array format: need to extract positive class
        shap_positive = np.array(shap_values)
    
    # Debug: print original shape
    original_shape = shap_positive.shape
    
    # Handle different shapes to extract positive class (class 1) as 1D vector
    if len(shap_positive.shape) == 1:
        # Already 1D - should be (n_features,)
        pass
    elif len(shap_positive.shape) == 2:
        if shap_positive.shape[1] == 2:
            # Shape (n_features, 2) -> take positive class column (index 1)
            shap_positive = shap_positive[:, 1]
        elif shap_positive.shape[0] == 1:
            # Shape (1, n_features) -> take first row
            shap_positive = shap_positive[0]
        else:
            # Unexpected 2D shape, flatten
            shap_positive = shap_positive.flatten()
    elif len(shap_positive.shape) == 3:
        # Shape (1, n_features, 2) -> take [0, :, 1] for positive class
        if shap_positive.shape[2] == 2:
            shap_positive = shap_positive[0, :, 1]
        else:
            shap_positive = shap_positive.flatten()
    else:
        # Higher dimensional, flatten
        shap_positive = shap_positive.flatten()
    
    # Final check: ensure it's 1D with shape (n_features,)
    shap_positive = np.array(shap_positive).flatten()
    
    # Verify shape is 1D - if not, force extraction
    if len(shap_positive.shape) != 1:
        # Last resort: if still not 1D, try to extract or flatten
        if len(shap_positive.shape) == 2 and shap_positive.shape[1] == 2:
            shap_positive = shap_positive[:, 1]
        else:
            shap_positive = shap_positive.flatten()
        shap_positive = np.array(shap_positive).flatten()
    
    # Final verification
    if len(shap_positive.shape) != 1:
        raise ValueError(f"SHAP values should be 1D, but got shape {shap_positive.shape} after processing. Original shape was {original_shape}")
    
    return shap_positive, explainer
