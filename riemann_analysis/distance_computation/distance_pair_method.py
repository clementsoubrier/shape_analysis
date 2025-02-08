import os
import numpy as np
import hashlib
from geomstats.geometry.discrete_curves import ElasticMetric
from geomstats.geometry.discrete_curves import DiscreteCurvesStartingAtOrigin
from .src.interpolation import interpolate, preprocess
from .src.alignment import align
from .src.projection import project_on_kendall_space
import geomstats.backend as gs

def calculate_riemann(data_dir, output_dir=None):
    riemann_distances, times, centroids = [], [], []
    a=1 
    b=0.5
    CURVES_SPACE_ELASTIC = DiscreteCurvesStartingAtOrigin(
        ambient_dim=2, k_sampling_points=1000, equip=False
    )
    CURVES_SPACE_ELASTIC.equip_with_metric(ElasticMetric, a=a, b=b)

    cells_path = os.path.join(data_dir, "cells")
    cell_dirs = [d for d in os.listdir(cells_path) if os.path.isdir(os.path.join(cells_path, d))]
    
    for cell_dir in cell_dirs:
        cell_i = int(cell_dir.split("_")[-1])
        cell_path = os.path.join(cells_path, cell_dir)
        number_of_frames = sum(
            os.path.isdir(os.path.join(cell_path, entry)) for entry in os.listdir(cell_path)
        )

        iter_distance = np.zeros(number_of_frames)
        iter_time = np.zeros(number_of_frames)
        iter_centroid = np.array([np.random.rand(2) for _ in range(number_of_frames)])
        
        BASE_LINE = np.load(os.path.join(cell_path, "frame_1/outline.npy"))
        BASE_LINE = interpolate(BASE_LINE, 1000)
        BASE_LINE = preprocess(BASE_LINE)
        
        for i in range(number_of_frames):
            frame_path = os.path.join(cell_path, f"frame_{i+1}")
            border_cell = np.load(os.path.join(frame_path, "outline.npy"))
            cell_interpolation = interpolate(border_cell, 1000)
            cell_preprocess = preprocess(cell_interpolation)
            border_cell = project_on_kendall_space(cell_interpolation)
            
            aligned_border = align(
                border_cell, BASE_LINE, rescale=True, rotation=False, reparameterization=True, k_sampling_points=1000
            )
            
            iter_distance[i] = CURVES_SPACE_ELASTIC.metric.dist(
                CURVES_SPACE_ELASTIC.projection(aligned_border), 
                CURVES_SPACE_ELASTIC.projection(BASE_LINE)
            )
            iter_time[i] = np.load(os.path.join(frame_path, "time.npy"))
            iter_centroid[i] = np.load(os.path.join(frame_path, "centroid.npy"))
            
            BASE_LINE = aligned_border

        riemann_distances.append(iter_distance)
        times.append(iter_time)
        centroids.append(iter_centroid)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        all_data = np.array([riemann_distances, times, centroids], dtype=object)
        hash_obj = hashlib.sha256()
        hash_obj.update(all_data.tobytes())
        hash_suffix = hash_obj.hexdigest()[:8]
        
        def save_data(filename, data):
            path = os.path.join(output_dir, f"{filename}_{hash_suffix}.npy")
            with open(path, 'wb') as f:
                np.save(f, np.array(data, dtype=object))
            return path
        
        riemann_path = save_data("riemann_distances", riemann_distances)
        times_path = save_data("times", times)
        centroids_path = save_data("centroids", centroids)
        
        return {
            "riemann_distances": riemann_path,
            "times": times_path,
            "centroids": centroids_path,
            "hash": hash_suffix
        }
    
    return None

calculate_riemann("", output_dir=None):
