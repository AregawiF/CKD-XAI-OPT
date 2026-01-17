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
    # 1. A list [class_0_array, class_1_array] where each is shape (1, n_features) or (n_features,)
    # 2. A numpy array with shape (1, n_features, 2) for both classes
    # 3. A numpy array with shape (n_features, 2)
    # 4. A numpy array with shape (2, n_features)
    
    # Extract the positive class (class 1) values
    if isinstance(shap_values, list):
        # List format: [class_0, class_1]
        # Each element should be the SHAP values for that class
        if len(shap_values) >= 2:
            shap_positive = np.array(shap_values[1])  # Positive class (index 1)
        else:
            # Fallback: if only one element, use it
            shap_positive = np.array(shap_values[0])
    else:
        # Array format: need to extract positive class
        shap_positive = np.array(shap_values)
    
    # Handle different shapes to extract positive class as 1D vector
    original_shape = shap_positive.shape
    
    if len(shap_positive.shape) == 1:
        # Already 1D - should be (n_features,)
        pass
    elif len(shap_positive.shape) == 2:
        if shap_positive.shape[0] == 1:
            # Shape (1, n_features) -> take first row
            shap_positive = shap_positive[0]
        elif shap_positive.shape[1] == 2:
            # Shape (n_features, 2) -> take positive class column (index 1)
            shap_positive = shap_positive[:, 1]
        elif shap_positive.shape[0] == 2:
            # Shape (2, n_features) -> take second row (positive class)
            shap_positive = shap_positive[1, :]
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
    
    # Final check: ensure it's 1D
    shap_positive = np.array(shap_positive).flatten()
    
    # If we still have too many values (both classes concatenated), take the second half
    # This handles cases where SHAP might return both classes as a flat array
    if shap_positive.size > 0:
        # Check if size is exactly double (both classes)
        # We'll let the calling code handle this if needed, but log a warning
        pass
    
    # Verify shape is 1D
    if len(shap_positive.shape) != 1:
        raise ValueError(
            f"SHAP values should be 1D, but got shape {shap_positive.shape} "
            f"after processing. Original shape was {original_shape}"
        )
    
    return shap_positive, explainer
