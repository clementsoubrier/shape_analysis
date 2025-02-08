import geomstats.backend as gs
import numpy as np 
#from numba import jit, njit, prange
import scipy.stats as stats
from scipy.integrate import simpson
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm



from geomstats.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOrigin,
    DynamicProgrammingAligner,
    ReparametrizationBundle,
    RotationBundle,
    ElasticMetric
)


def del_arr_elements(arr, indices):
    """
    Delete elements in indices from array arr
    """

    # Sort the indices in reverse order to avoid index shifting during deletion
    indices.sort(reverse=True)

    # Iterate over each index in the list of indices
    for index in indices:
        del arr[index]
    return arr





# def exhaustive_align(curve, ref_curve, k_sampling_points, rescale=True, dynamic=False, reparameterization=True):
#     """ 
#     Quotient out
#         - translation (move curve to start at the origin) 
#         - rescaling (normalize to have length one)
#         - rotation (try different starting points, during alignment)
#         - reparametrization (resampling in the discrete case, during alignment)
    
#     :param bool rescale: quotient out rescaling or not 
#     :param bool dynamic: Use dynamic aligner or not 
#     :param bool reparamterization: quotient out rotation only rather than rotation and reparameterization

#     """
    
#     curves_r2 = DiscreteCurvesStartingAtOrigin(
#         ambient_dim=2, k_sampling_points=k_sampling_points, equip=False
#     )

#     if dynamic:
#         print("Use dynamic programming aligner")
#         curves_r2.fiber_bundle = ReparametrizationBundle(curves_r2)
#         curves_r2.fiber_bundle.aligner = DynamicProgrammingAligner()

#     # Quotient out translation
#     print("Quotienting out translation")
#     curve = curves_r2.projection(curve)
#     ref_curve = curves_r2.projection(ref_curve)

#     # Quotient out rescaling
#     if rescale:
#         print("Quotienting out rescaling")
#         curve = curves_r2.normalize(curve)
#         ref_curve = curves_r2.normalize(ref_curve)

#     # Quotient out rotation and reparamterization
#     curves_r2.equip_with_metric(ElasticMetric)
#     if not reparameterization:
#         print("Quotienting out rotation")
#         curves_r2.equip_with_group_action("rotations")
#     else:
#         print("Quotienting out rotation and reparamterization")
#         curves_r2.equip_with_group_action("rotations and reparametrizations")
        
#     curves_r2.equip_with_quotient_structure()
#     aligned_curve = curves_r2.fiber_bundle.align(curve, ref_curve)
#     return aligned_curve




def rotation_align(curve, base_curve, k_sampling_points):
    """Align curve to base_curve to minimize the L² distance by \
        trying different start points.

    Returns
    -------
    aligned_curve : discrete curve
    """
    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    base_curve = gs.array(base_curve)

    # Rotation is done after projection, so the origin is removed
    total_space = DiscreteCurvesStartingAtOrigin(k_sampling_points=k_sampling_points-1)
    total_space.fiber_bundle = RotationBundle(total_space)

    for shift in range(nb_sampling):
        reparametrized = [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        aligned = total_space.fiber_bundle.align(
            point=gs.array(reparametrized), base_point=base_curve
        )
        distances[shift] = np.linalg.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)
    reparametrized_min = [
        curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
    ]
    aligned_curve = total_space.fiber_bundle.align(
        point=gs.array(reparametrized_min), base_point=base_curve
    )
    return aligned_curve


def align(point, base_point, rescale, rotation, reparameterization, k_sampling_points):
    """
    Align point and base_point via quotienting out translation, rescaling, rotation and reparameterization
    """

    total_space = DiscreteCurvesStartingAtOrigin(k_sampling_points=k_sampling_points)
   
    
    # Quotient out translation 
    point = total_space.projection(point) 
    point = point - gs.mean(point, axis=0)

    base_point = total_space.projection(base_point)
    base_point = base_point - gs.mean(base_point, axis=0)

    # Quotient out rescaling
    if rescale:
        point = total_space.normalize(point) 
        base_point = total_space.normalize(base_point)
    
    # Quotient out rotation
    if rotation:
        point = rotation_align(point, base_point, k_sampling_points)

    # Quotient out reparameterization
    if reparameterization:
        aligner = DynamicProgrammingAligner(total_space)
        total_space.fiber_bundle = ReparametrizationBundle(total_space, aligner=aligner)
        point = total_space.fiber_bundle.align(point, base_point)
    return point