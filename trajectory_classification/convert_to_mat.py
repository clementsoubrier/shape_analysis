import numpy as np
import hashlib
from scipy.io import savemat

def save_trajectory_to_mat(centroids_path, output_base="trajectory_data_segm"):
    centroids = np.load(centroids_path, allow_pickle=True)
    
    tracks = {}
    for i, trajectory in enumerate(centroids):
        n_frames = trajectory.shape[0]
        row = np.zeros(n_frames * 8)
        for j, (x, y) in enumerate(trajectory):
            start_idx = j * 8  
            row[start_idx] = x  
            row[start_idx + 1] = y  

        tracks[f"track_{i+1}"] = row  

    temp_output = f"{output_base}.mat"
    savemat(temp_output, {'tracks': tracks})
    
    sha256_hash = hashlib.sha256()
    with open(temp_output, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    hash_suffix = sha256_hash.hexdigest()[:8]
    output_path = f"{output_base}_{hash_suffix}.mat"
    
    savemat(output_path, {'tracks': tracks})
    
    return output_path, sha256_hash.hexdigest()

hash_path, hash_value = save_trajectory_to_mat(
    ""
)
