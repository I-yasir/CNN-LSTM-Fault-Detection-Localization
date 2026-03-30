import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_dataset(collected_data_path , processed_data_path):

    data = np.load(str(collected_data_path))
    # print(data.files)

    X = data['signals_VI']
    y_class = data['fault_class']
    y_dist = data['fault_dist']

    N,T,F = X.shape

    X_reshaped = X.reshape(-1,F)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    X_final = X_scaled.reshape(N, T, F)

    # print(f"Mean after scaling: {X_final.mean():.4f}") # Should be ~0
    # print(f"Std after scaling: {X_final.std():.4f}")   # Should be ~1

    np.savez(processed_data_path , signals_VI = X_final, fault_class = y_class, fault_dist = y_dist)