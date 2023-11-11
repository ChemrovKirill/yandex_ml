import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """

    def norm2(v):
        return np.sqrt(v.T @ v)
    
    r = np.random.random(data.shape[0])
    for _ in range(num_steps):
        prod = data @ r
        r = prod / norm2(prod)
    eigval = r.T @ (data @ r) / norm2(r)
    
    return float(eigval), r
