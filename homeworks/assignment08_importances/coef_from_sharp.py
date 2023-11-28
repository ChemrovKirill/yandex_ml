import numpy as np
def get_coef_from_shap_values(shap_values, X_train_scaled):
    x_len = X_train_scaled.shape[0]
    if x_len > 100:
        np.random.seed(0)
        inds = np.random.choice(x_len, size=100, replace=False)
    else:
        inds = range(x_len)
    return np.mean(shap_values.values * (X_train_scaled - X_train_scaled[inds].mean(0, keepdims=True)), axis=0)

    

