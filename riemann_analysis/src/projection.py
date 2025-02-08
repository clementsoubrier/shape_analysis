import numpy as np
from geomstats.geometry.pre_shape import PreShapeSpace
import matplotlib.pyplot as plt

import geomstats.backend as gs


def project_on_kendall_space(curve,PRESHAPE_SPACE= None):
    if PRESHAPE_SPACE is None:
        PRESHAPE_SPACE = PreShapeSpace(ambient_dim=2, k_landmarks=len(curve))
    projected_curve = PRESHAPE_SPACE.projection(curve)
    return projected_curve
