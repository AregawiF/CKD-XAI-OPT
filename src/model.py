from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_rf(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=5, random_state=42):
    """
    Train Random Forest classifier and evaluate on test set.
    Returns the trained model and metrics.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Random Forest Test Accuracy: {acc:.4f}")
    print(f"Random Forest Test F1-Score: {f1:.4f}")
    
    return rf, acc, f1
