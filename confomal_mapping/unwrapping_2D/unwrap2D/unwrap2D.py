# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:15:36 2022

@author: fyz11
"""

def transform_contour(contour, tform_matrix):
    
    import numpy as np 
    
    contour_out = tform_matrix.dot(np.vstack([contour.T, 
                                              np.ones(len(contour))]))[:2].T
    
    return contour_out


def pixelize_contours(contour, imshape=None, padsize=50):
    
    import numpy as np 
    import skimage.draw as skdraw
    

    if imshape is None:
        m_max = np.max(contour[...,0])
        n_max = np.max(contour[...,1])
        
        m_max = m_max + padsize
        n_max = n_max + padsize
        
    else:
        m, n = imshape[:2]
        
        m_max = m 
        n_max = n 
    blank = np.zeros((m_max,n_max))

    yy,xx = skdraw.polygon(contour[:,0], 
                           contour[:,1], 
                           shape=imshape)

    blank[yy,xx] = 1

    return blank    
    

def register_contour_array(contour_array, ref_id=0, tform='Similarity'):
    
    import skimage.transform as sktform 
    import numpy as np 
    
    # register: 
    if tform == 'Similarity':
        tform = sktform.SimilarityTransform()
    if tform == 'Euclidean':
        tform = sktform.EuclideanTransform()
    if tform == 'Affine':
        tform = sktform.AffineTransform()
        

    contour_array_out = np.zeros_like(contour_array)
    
    for iii in np.arange(contour_array.shape[-1]):
        
        cnt = contour_array[...,iii].copy()
        tform.estimate(contour_array[...,iii], 
                       contour_array[...,ref_id]) # dst 
        
        cnt_tfm = transform_contour(cnt, tform.params)
        contour_array_out[...,iii] = cnt_tfm.copy()
        
    return contour_array_out


def contour_area(contour, close_contour=True):
    
    import numpy as np 
    contour_ = contour.copy()
    if close_contour:
        contour_ = np.vstack([contour_, 
                              contour_[0][None,:]]).T
    
    x=contour_[:,0]
    y=contour_[:,1]
    area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
    area=np.abs(area)
    
    return area 

# construct the laplacian grid for a 2D image. (might be faster to exploit kron! )
def _laplacian_matrix(n, m, mask=None): # this is the only working solution! 
    
    import scipy.sparse
    from scipy.sparse.linalg import spsolve
    import numpy as np 
    
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
    
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    if mask is not None:
        y_range = mask.shape[0]
        x_range = mask.shape[1]
        # find the masked i.e. zeros
        zeros = np.argwhere(mask==0) # in (y,x)
        k = zeros[:,1] + zeros[:,0] * x_range
        mat_A[k,k] = 1
        mat_A[k, k + 1] = 0
        mat_A[k, k - 1] = 0
        mat_A[k, k + x_range] = 0
        mat_A[k, k - x_range] = 0
        
        mat_A = mat_A.tocsc()
    else:
        mat_A = mat_A.tocsc()
    
    return mat_A

def poisson_dist_tform(binary, pt=None):
    """
    Computation for a single binary image. 
    """
    import scipy.sparse
    from scipy.sparse.linalg import spsolve
    import numpy as np 
    
    mask = np.pad(binary, [1,1], mode='constant', constant_values=1) # pad with ones.  # ones need for connectivity....
    
    mat_A = _laplacian_matrix(mask.shape[0], mask.shape[1], mask=mask) # this is correct!

    if pt is not None:
        mask = np.zeros_like(mask)
        mask[pt[:,0], pt[:,1]] = 1
        
        mask_flat = mask.flatten()    
        mat_b = np.zeros(len(mask_flat))
        mat_b[mask_flat == 1] = 1
        
    else:
        # now we need to zero the padding
        mask[0,:]=0
        mask[:,-1]=0
        mask[-1,:]=0
        mask[:,0]=0
        
        # solve within the mask!    
        mask_flat = mask.flatten()    
        # inside the mask:
        # \Delta f = div v = \Delta g       
        mat_b = np.ones(len(mask_flat)) # does this matter -> the amount... 
        mat_b[mask_flat == 0] = 0
        
    x = spsolve(mat_A, mat_b, permc_spec='MMD_AT_PLUS_A') 
    x = x.reshape(mask.shape)
    x = x[1:-1,1:-1].copy()
    x = x - x.min() # solution is only positive!.
    
    return x 


def sdf_poisson(mask, pt=None):
    
    import numpy as np 
    import scipy.ndimage as ndimage
    import skimage.morphology as skmorph
    import pylab as plt 
    # import unwrap3D.Segmentation.segmentation as unwrap3D_segmentation 
    # all_ones = np.ones(mask.shape, dtype=bool)
    border_pts = np.zeros(mask.shape, dtype=bool)
    border_pts[:1] = 1
    border_pts[-1:] = 1
    border_pts[:,:1] = 1
    border_pts[:,-1:] = 1
    border_coords = np.argwhere(border_pts>0)
    
    # print(border_coords.shape)
    
    sdf = np.zeros(mask.shape[:2]) 
    sdf_inner = poisson_dist_tform(mask, pt=pt)
    # sdf_outer = ndimage.distance_transform_edt(np.logical_not(mask>0))
    sdf_outer = poisson_dist_tform(np.logical_not(mask), pt=border_coords)
    # plt.figure()
    # plt.imshow(sdf_outer)
    # plt.show()
    sdf[np.logical_not(mask>0)] = -sdf_outer[np.logical_not(mask>0)]
    sdf[mask>0] = sdf_inner[mask>0]
    
    return sdf 


def measure_signed_distance(contour_ref, contours, imshape):
    
    import skimage.draw as skdraw
    import numpy as np 
    
    mask = np.zeros(imshape, dtype=bool)
    yy,xx = skdraw.polygon(contour_ref[:,0], 
                           contour_ref[:,1], shape=imshape)
    mask[yy,xx] = 1
    
    sign_dists = mask[contours[0,:,0].astype(np.int32), 
                      contours[0,:,1].astype(np.int32)]==0
    sign_dists = sign_dists*1
    sign_dists[sign_dists==0] = -1
    
    dists = np.nansum(np.linalg.norm(np.diff(contours, axis=0), axis=-1), axis=0) * sign_dists
    
    return dists, sign_dists
    

def map_contour_to_unit_circle(pts, close_contour = True):
    
    import numpy as np 
    
    pts_ = pts.copy()
    
    if close_contour: 
        pts_ = np.vstack([pts_, 
                          pts_[0][None,:]])
    
    pts_dists = np.linalg.norm(np.diff(pts_, axis=0), axis=-1)
    pts_dists = np.hstack([0, np.cumsum(pts_dists)])    
    # rescale the cum differences. to be circumference of circle i.e. 2*np.pi
    
    theta = pts_dists/float(pts_dists[-1]) * 2*np.pi
    
    theta_coords = np.vstack([np.cos(theta), 
                              np.sin(theta)]).T

    if close_contour:
        theta_coords = theta_coords[:-1]
    
    return theta_coords
    
    
def map_protrusion_topography_to_circle(contour_ref, signed_protrusion_d, circle_R=50, close_contour=True):
    
    # map contour to circle
    circle_coords = map_contour_to_unit_circle(contour_ref, close_contour = close_contour)
    
    normal = circle_coords.copy()
    circle_coords = circle_R*circle_coords
    
    # now offset the normal 
    topo_coords = circle_coords + signed_protrusion_d[:,None] * normal
    
    return topo_coords, circle_coords
    


def curvature_splines(x, y, k=4, error=0.1):
    """Calculate the signed curvature of a 2D curve at each point using interpolating splines.
    
    Parameters
    ----------
    x : (n_points,) array 
        x-coordinate of the contour 
    y : (n_points,) array 
        y-coordinate of the contour 
    k : int
        order of the interpolating spline
    error : float
        The admisible error when interpolating the splines
        
    Returns
    -------
    [x_, y_] : [(n_points,) array, (n_points,) array]
        the spline evaluated (and smoothened) x and y coordinate at each point of a 2D curve for which curvature is evaluated at
    [x_prime, y_prime] : [(n_points,) array, (n_points,) array]
        the 1st derivative of the x and y coordinate at each point of a 2D curve
    curvature : (n_points,) array 
        the signed curvature of a 2D curve at each point 
    """
    from scipy.interpolate import UnivariateSpline
    import numpy as np 
    
    # handle list of complex case
    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=k, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=k, w=1 / np.sqrt(std))

    x_ = fx(t)
    y_ = fy(t)

    x_prime = fx.derivative(1)(t)
    x_prime_prime = fx.derivative(2)(t)
    y_prime = fy.derivative(1)(t)
    y_prime_prime = fy.derivative(2)(t)
    curvature = (x_prime * y_prime_prime - y_prime* x_prime_prime) / np.power(x_prime** 2 + y_prime** 2, 3. / 2)
#    return [x_, y_], [xˈ, yˈ], curvature
    return [x_, y_], [x_prime, y_prime], curvature



def discrete_curvature(x,y):
    
    def _PJcurvature(x,y):
        """
        input  : the coordinate of the three point
        output : the curvature and norm direction
        """
        import numpy as np
        import numpy.linalg as LA

        t_a = LA.norm([x[1]-x[0],y[1]-y[0]])
        t_b = LA.norm([x[2]-x[1],y[2]-y[1]])
        
        M = np.array([
            [1, -t_a, t_a**2],
            [1, 0,    0     ],
            [1,  t_b, t_b**2]
        ])
    
        a = np.matmul(LA.inv(M),x)
        b = np.matmul(LA.inv(M),y)
    
        kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
        return kappa, [b[1],-a[1]]/np.sqrt(a[1]**2.+b[1]**2.)
        
    import numpy as np 
    
    x_t = np.hstack([x[-1],
                     x, 
                     x[0]])
    y_t = np.hstack([y[-1], 
                     y,
                     y[0]])
    
    x_y_t = np.vstack([x_t, y_t]).T
    curve_t = []
    
    for idx, xy in enumerate(x_y_t[1:-1]):
        x = x_y_t[idx:idx+3,0]
        y = x_y_t[idx:idx+3,1]
        kappa,norm = _PJcurvature(x,y)
        curve_t.append(kappa)
        
    return np.hstack(curve_t)
        

def baseline_als(y, lam, p, niter=10):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    import numpy as np 
    
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def resample_disk_grid_to_img_grid(disk, img_pts, faces, img_I, raster_size=256, border_pad=32):
    
    """
    This might have clipped the boundaries.... 
    """
    import numpy as np  
    import unwrap3D.Mesh.meshtools as unwrap3D_meshtools
    import unwrap3D.Image_Functions.image as unwrap3D_image_fn
    
    disk_pts = (np.clip(disk, -1,1.) + 1)/2.*raster_size + border_pad
    
    N = raster_size+2*border_pad
    unwrap_params = np.zeros((N, N, img_pts.shape[-1]))
    YY, XX = np.indices(unwrap_params.shape[:2])
    mask = np.sqrt((XX-N/2.)**2 + (YY-N/2.)**2) <= raster_size/2.
    
    mask_img_out = np.vstack([YY[mask>0], 
                              XX[mask>0]]).T
    
    mesh_ref = unwrap3D_meshtools.create_mesh(np.hstack([np.zeros(len(disk_pts))[:,None], 
                                                  disk_pts]), 
                                                   faces)
    mesh_match = unwrap3D_meshtools.match_and_interpolate_uv_surface_to_mesh(np.hstack([np.zeros(len(mask_img_out))[:,None], 
                                                                                        mask_img_out]).reshape(-1,3), 
                                                                             mesh_ref, match_method='cross')
    
    # mask_img_out_2D_img_pts = np.array([mesh_vertex_interpolate_scalar(mesh_ref, 
    #                                                                    mesh_match[0], 
    #                                                                    mesh_match[1], 
    #                                                                    img_pts[...,ch].ravel()) for ch in np.arange(img_pts.shape[-1])]).T
    mask_img_out_2D_img_pts = unwrap3D_meshtools.mesh_vertex_interpolate_scalar(mesh_ref, 
                                                                       mesh_match[0], 
                                                                       mesh_match[1],
                                                                       img_pts)
    unwrap_params[mask>0] = mask_img_out_2D_img_pts.copy()
    
    
    unwrap_img = np.zeros((N, N))
    I_verts = unwrap3D_image_fn.map_intensity_interp2(mask_img_out_2D_img_pts, 
                                                      grid_shape=img_I.shape, 
                                                      I_ref=img_I)
    unwrap_img[mask>0] = I_verts.copy()
    
    return unwrap_params, unwrap_img, mask>0 # this is the mapping already... 
    


def area_distortion_flow_relax_disk(mesh, mesh_orig, 
                                    max_iter=50,
                                    # smooth_iters=5,  
                                    delta_h_bound=0.5, 
                                    stepsize=.1, 
                                    adaptive_step=False, # not yet implemented   
                                    flip_delaunay=True, # do this in order to dramatically improve flow!. 
                                    robust_L=False, # use the robust laplacian instead of cotmatrix - slows flow. 
                                    mollify_factor=1e-5,
                                    eps = 1e-8,
                                    lam = 1e-4, 
                                    debugviz=False,
                                    debugviz_tri=True):
    
    """
    implements the solution of 
    1) Zou et al. https://ieeexplore.ieee.org/document/6064964 - Authalic parametrization of general surfaces using Lie advection 
    on a disk -> we can then map to square with any number of methods. 
    # 2) Su et al. https://reader.elsevier.com/reader/sd/pii/S0167839619300524?token=3C2D892CEFA4E8360DAAAF8E08CBAC4A090E7DB9DF61AFDA2006D2B54C1360F43030B217C788180F87E1B3157C76EB07&originRegion=us-east-1&originCreation=20220706183843
    """
    import igl 
    import numpy as np 
    import scipy.sparse as spsparse
    from tqdm import tqdm 
    import unwrap3D.Mesh.meshtools as unwrap3D_meshtools
    
    # import meshplex
    import pylab as plt
    if robust_L:
        import robust_laplacian

    V = mesh.vertices.copy()
    F = mesh.faces.copy()

    v = mesh.vertices.copy()
    f = mesh.faces.copy()
    
    try:
        f = igl.intrinsic_delaunay_triangulation(igl.edge_lengths(v,f), f)[1]
    except:
        pass

    if robust_L:
        L, M = robust_laplacian.mesh_laplacian(np.array(v), np.array(f), mollify_factor=mollify_factor)
    else:
        L = -igl.cotmatrix(v,f)

    area_distortion_iter = []
    v_steps = [v]
    f_steps = [f]

    for ii in tqdm(range(max_iter)):
        
        try:
            # if conformalize == False:
            if robust_L:
                L, m = robust_laplacian.mesh_laplacian(np.array(v), np.array(f),mollify_factor=mollify_factor); # this must be computed. # if not... then no growth -> flow must change triangle shape!. 
            else:
                L = -igl.cotmatrix(v,f)
    
            v_bound = igl.boundary_loop(f)
                    
            # # compute the area distortion of the face -> having normalized for surface area. -> this is because the sphere minimise the surface area. -> guaranteeing positive. 
            A2 = igl.doublearea(mesh_orig.vertices/np.sqrt(np.nansum(igl.doublearea(mesh_orig.vertices,f)*.5)), f)
            A1 = igl.doublearea(v/np.sqrt(np.nansum(igl.doublearea(v,f)*.5)), f)
            
            # B = np.log10(A1/(A2)) # - 1
            B = (A1+eps)/(A2+eps) - 1 # adding regularizer to top and bottom is better!. 
    
            area_distortion_mesh = (A2/(A1+eps)) # this is face measure!. and the connectivity is allowed to change! during evolution !. 
            area_distortion_mesh_vertex = igl.average_onto_vertices(v, 
                                                                    f, 
                                                                    np.vstack([area_distortion_mesh,area_distortion_mesh,area_distortion_mesh]).T)[:,0]
    
            # smooth ... 
            if debugviz:
                plt.figure()
                plt.hist(np.log10(area_distortion_mesh_vertex)) # why no change? 
                plt.show()
    
            # if smooth_iters > 0:
            #     smooth_area_distortion_mesh_vertex = np.vstack([area_distortion_mesh_vertex,area_distortion_mesh_vertex,area_distortion_mesh_vertex]).T # smooth this instead of the gradient. 
    
            #     for iter_ii in range(smooth_iters):
            #         smooth_area_distortion_mesh_vertex = igl.per_vertex_attribute_smoothing(smooth_area_distortion_mesh_vertex, f) # seems to work.
            #     area_distortion_mesh_vertex = smooth_area_distortion_mesh_vertex[:,0]
    
            B = np.clip(B, -delta_h_bound, delta_h_bound) # bound above and below. 
            # B_vertex = igl.average_onto_vertices(v, 
            #                                       f, 
            #                                       np.vstack([B,B,B]).T)[:,0] # more accurate? 
            B_vertex = unwrap3D_meshtools.f2v(v,f).dot(B)
    
            # from scipy.sparse.linalg import lsqr ---- this sometimes fails!... 
            I = spsparse.spdiags(lam*np.ones(len(v)), [0], len(v), len(v)) # tikholov regulariser. 
            g = spsparse.linalg.spsolve((L.T).dot(L) + I, (L.T).dot(B_vertex)) # solve for a smooth potential field.  # this is the least means square. 
            # g = spsparse.linalg.lsqr(L.T.dot(L), L.dot(B_vertex), iter_lim=100)[0] # is there a better way to solve this quadratic? 
    
            face_vertex = v[f].copy()
            face_normals = np.cross(face_vertex[:,1]-face_vertex[:,0], 
                                    face_vertex[:,2]-face_vertex[:,0], axis=-1)
            face_normals = face_normals / (np.linalg.norm(face_normals, axis=-1)[:,None] + eps)
            # face_normals = np.vstack([np.ones(len(face_vertex)), 
            #                           np.zeros(len(face_vertex)),
            #                           np.zeros(len(face_vertex))]).T # should this be something else? 
            face_g = g[f].copy()
            
            # vertex_normals = igl.per_vertex_normals(v,f)
    
            # i,j,k = 1,2,3
            face_vertex_lhs = np.concatenate([(face_vertex[:,1]-face_vertex[:,0])[:,None,:],
                                              (face_vertex[:,2]-face_vertex[:,1])[:,None,:],
                                              face_normals[:,None,:]], axis=1) 
            face_g_rhs = np.vstack([(face_g[:,1]-face_g[:,0]),
                                    (face_g[:,2]-face_g[:,1]),
                                     np.zeros(len(face_g))]).T
    
    
            # solve a simultaneous set of 3x3 problems
            dg_face = np.linalg.solve( face_vertex_lhs, face_g_rhs)
    
            gu_mag = np.linalg.norm(dg_face, axis=1) 
            max_size = igl.avg_edge_length(v, f) / np.nanmax(gu_mag) # stable if divide by nanmax # must be nanmax!. 
            
            # dg_face = stepsize*max_size*dg_face # this is vector. and is scaled by step size 
            # dg_face = max_size*dg_face
            
            # average onto the vertex. 
            dg_vertex = igl.average_onto_vertices(v, 
                                                  f, 
                                                  dg_face)
            dg_vertex = dg_vertex * max_size
            # dg_vertex = dg_vertex - np.nansum(dg_vertex*vertex_normals,axis=-1)[:,None]*vertex_normals
            
            # correct the flow at the boundary!. # this is good? ---> this is good for an explicit euler. 
            normal_vect = L.dot(v)
            normal_vect = normal_vect / (np.linalg.norm(normal_vect, axis=-1)[:,None] + 1e-8)
    
            # dg_vertex[v_bound] = dg_vertex[v_bound] - np.nansum(dg_vertex[v_bound] * v[v_bound], axis=-1)[:,None]*v[v_bound]
    
            """
            this is the gradient at the vertex. 
            """
            dg_vertex[v_bound] = dg_vertex[v_bound] - np.nansum(dg_vertex[v_bound] * normal_vect[v_bound], axis=-1)[:,None]*normal_vect[v_bound]
            # disps.append(dg_vertex)
            
            """
            Adaptive step length based on minimal triangular collapse. 
                from https://par.nsf.gov/servlets/purl/10185275 Discrete Lie Flow paper. 
            """
            if adaptive_step:
                """
                This doesn't work!. 
                """
                dH = dg_vertex[:,:][f].copy(); #dH = dH*(max_size)
                P1 = v[:,:][f].copy() # get the triangles 1 and just the 2D coordinates!
                # P1 = P1[...,1:].copy()
                print(dH.max(), dH.min())
                # # isometric projection of triangles to plane, first.  
                # [U,UF,I] = igl.project_isometrically_to_plane(v,f) # V to U.    
                # # # from this we can find the rotation. p3 x R = p2... (3x3) (3x2) = (3x2) 
                # R3_to_2D = np.matmul(np.linalg.inv(v[f]), U[UF])
                
                # P1 = U[UF] # (x1,x2)
                # # P2 = np.matmul((v+vA_vertex/(np.linalg.norm(vA_vertex,axis=-1).max()))[f], R3_to_2D) 
                # P2 = np.matmul((v+vA_vertex)[f], R3_to_2D) 
                # dH = P2 - P1  # dH # this is the projected 2D warp. 
                # # determine the triangle flipping ?
                
                x1 = P1[:,0,1].copy(); y1 = P1[:,0,2].copy()
                x2 = P1[:,1,1].copy(); y2 = P1[:,1,2].copy()
                x3 = P1[:,2,1].copy(); y3 = P1[:,2,2].copy()
                
                h11 = dH[:,0,1].copy(); h12 = dH[:,0,2].copy()
                h21 = dH[:,1,1].copy(); h22 = dH[:,1,2].copy()
                h31 = dH[:,2,1].copy(); h32 = dH[:,2,2].copy()
                
                Delta = x2*h12 - x1*h22 + h31*(h22-h12) - h32*(h21-h11)
                S = h31*(y2-y1) + x3*(h22-h12) - h32*(x2-x1) - y3*(h12-h11+y2*(h21-h11))
                Q = x2*y1-x1*y2-y3*(x2-x1)
                
                # Delta = -Delta
                # S = -S
                # Q= -Q
                
                # print(Delta.max(), S.max(), Q.max())
                # print(Delta.min(), S.min(), Q.min())
                
                select = np.abs(Delta) <= eps
                nonselect = np.logical_not(select)
                select_ids = np.arange(len(Delta))[select]
                nonselect_ids = np.arange(len(Delta))[nonselect]
    
                lam1 = (-S + np.sqrt(S**2 - 4*Delta*Q)) / (2*Delta + eps)
                lam2 = (-S - np.sqrt(S**2 - 4*Delta*Q)) / (2*Delta + eps)
                
                select_lams = (-Q/S)[select_ids]
                nonselect_lams1 = lam1[nonselect_ids]
                nonselect_lams2 = lam2[nonselect_ids] 
    
                all_lams = np.hstack([select_lams, nonselect_lams1, nonselect_lams2])
                
                all_lams_pos = all_lams > 0 
                # print(all_lams.min(), all_lams.max())
    
                try: 
                    min_lam = np.nanmin(all_lams[all_lams_pos])
                except:
                    min_lam = stepsize
    
                if ii == 0:
                    base_lam = min_lam
                    
                scale_factor = min_lam / base_lam * 0.8 # 
                # print(min_lam, scale_factor)
            else:
                # print('no_adaptive')
                scale_factor=stepsize
            # print('scale_factor, ', scale_factor)
            """
            advection step 
            """
            # v = V[:,1:] + np.array(disps).sum(axis=0)[:,1:] # how to make this step stable? 
            v = v_steps[-1][:,1:] + scale_factor*dg_vertex[:,1:] # last one. 
            v = np.hstack([np.zeros(len(v))[:,None], v])
            
            if flip_delaunay: # we have to flip!. 
                # import meshplex
                # # this clears out the overlapping. is this necessary
                # mesh_out = meshplex.MeshTri(v, f)
                # mesh_out.flip_until_delaunay()
            
                # # update v and f!
                # v = mesh_out.points.copy()
                # f = mesh_out.cells('points').copy() 
                f = igl.intrinsic_delaunay_triangulation(igl.edge_lengths(v,f), f)[1]
                
                if np.sum(np.isnan(f)) > 0:
                    break
    
            if debugviz_tri:
                plt.figure(figsize=(5,5))
                plt.triplot(v[:,1],
                            v[:,2], f, 'g-', lw=.1)
                plt.show()
                
            v_steps.append(v)
            f_steps.append(f)
            area_distortion_iter.append(area_distortion_mesh) # append this. 

        except:
            # if error then break
            return v_steps, f_steps, area_distortion_iter

    return v_steps, f_steps, area_distortion_iter


def parametric_line_flow_2D(contour_pts,
                            external_img_gradient, 
                            E=None, 
                            close_contour=True, 
                            fixed_boundary = False, 
                            lambda_flow=1000, 
                            step_size=1,
                            niters=10, 
                            conformalize=True,
                            eps=1e-12):

    import numpy as np 
    import scipy.sparse as spsparse
    # from ..Unzipping import unzip_new as uzip 
    import unwrap3D.Mesh.meshtools as unwrap3D_meshtools
    import unwrap3D.Image_Functions.image as unwrap3D_imagefn

    if E is None:
        if close_contour:
            E = [np.arange(len(contour_pts)), 
                 np.hstack([np.arange(len(contour_pts))[1:], 0])]
        else:
            E = [np.arange(len(contour_pts))[:-1], 
                 np.arange(len(contour_pts))[1:]]
        E = np.vstack(E).T

    A = unwrap3D_meshtools.adjacency_edge_cost_matrix(contour_pts, E)
    L = A-spsparse.diags(np.squeeze(np.array(A.sum(axis=1)))); # why is this so slow?  #### is this correct? 

    if fixed_boundary:
        boundary_nodes = np.arange(len(contour_pts))[A.sum(axis=1) == 1]
        L[boundary_nodes,:] = 0 # slice this in. 
    
    # so we need no flux boundary conditions to prevent flow in x,y at the boundary!....----> one way is to do mirror...( with rectangular grid this is easy... but with triangle is harder...)
    contour_pts_flow = [contour_pts]
    for iter_ii in np.arange(niters):
        A = unwrap3D_meshtools.adjacency_edge_cost_matrix(contour_pts_flow[-1], E)
        if conformalize==False:
            L = A-spsparse.diags(np.squeeze(np.array(A.sum(axis=1))));
            if fixed_boundary:
                boundary_nodes = np.arange(len(contour_pts))[A.sum(axis=1) == 1]
                L[boundary_nodes,:] = 0 # slice this in. 
        M = unwrap3D_meshtools.mass_matrix2D(A)
        
        """
        look up external gradients. 
        """
        U_grad = np.array([unwrap3D_imagefn.map_intensity_interp2(contour_pts_flow[-1], 
                                                      grid_shape=external_img_gradient.shape[:-1], 
                                                      I_ref=external_img_gradient[...,ch]) 
                                                       for ch in np.arange(external_img_gradient.shape[-1])])
        U_grad = U_grad.T
        U_grad = U_grad / (np.linalg.norm(U_grad, axis=-1)[:,None] + eps)
        
        vvv = spsparse.linalg.spsolve(M-lambda_flow*L, M.dot(contour_pts_flow[-1] + U_grad * step_size))
        
        contour_pts_flow.append(vvv)
    contour_pts_flow = np.array(contour_pts_flow)
    contour_pts_flow = contour_pts_flow.transpose(1,2,0)

    return contour_pts_flow



# Master function for unwrapping given 2D image and binary mask
def unwrap_2D(img, 
              mask, 
              conformal_map=False, 
              relax_tol=1.0e-5,
              relax_niters=25,
              relax_omega=1.,
              debugviz_tri_areadistort = False,
              areadistort_max_iter = 100,
              areadistort_delta_h_bound = 0.5,
              areadistort_stepsize=0.1,
              area_distort_flip_tri=True): 
    
    import numpy as np 
    import igl 
    import unwrap3D.Mesh.meshtools as unwrap3D_meshtools
    
    # 1. build a mesh from the binary 
    grid_quads, grid_tri = unwrap3D_meshtools.get_uv_grid_quad_connectivity(mask>0, 
                                                                            return_triangles=True, 
                                                                            bounds='none')
    
    grid_pts = np.dstack(np.indices(mask.shape)).reshape(-1,2)
    grid_pts = np.hstack([np.zeros(len(grid_pts))[:,None], 
                          grid_pts])
    grid_bool = mask.ravel() > 0
    
    
    face_bool = grid_bool[grid_tri].copy()
    face_invalid_index = np.unique(np.argwhere(face_bool==0)[:,0])
    face_keep_index = np.setdiff1d(np.arange(len(grid_tri)), face_invalid_index)
    # keep_tri_indices = np.hstack([iii for iii in np.arange(len(grid_tri)) if np.sum(np.intersect1d(grid_tri, invalid_pts))==0])
    
    mesh_2D = unwrap3D_meshtools.create_mesh(grid_pts[:,:], grid_tri[face_keep_index][:,::-1])
    mesh_2D_comps = mesh_2D.split(only_watertight=False)
    mesh_2D_submesh = mesh_2D_comps[np.argmax([len(ccc.vertices) for ccc in mesh_2D_comps])]
    
    # 2. harmonic (conformal) mapping to the 2D disk 
    # disk_coords = rectangular_conformal_map(mesh_2D_submesh.vertices,
    #                                         mesh_2D_submesh.faces[:,:],
    #                                         corner=None,  
    #                                         random_state=0)
    disk_coords = unwrap3D_meshtools.rectangular_conformal_map(mesh_2D_submesh.vertices,
                                                               mesh_2D_submesh.faces[:,:],
                                                               corner=None,  
                                                               random_state=0)
    
    # 3. resample the harmonic (conformal) mapping to the 2D disk to get better mesh quality 
    disk_coords_mesh = unwrap3D_meshtools.create_mesh(disk_coords, mesh_2D_submesh.faces)
    disk_coords_relax = [disk_coords_mesh]
    # disk_coords_relax = relax_mesh( disk_coords_mesh, 
    #                                 relax_method='CVT (block-diagonal)', 
    #                                 tol=relax_tol, 
    #                                 n_iters=relax_niters, 
    #                                 omega=relax_omega)
    
    # if conformal_map == False:
    
    # 4. Match and compute the corresponding real image coordinates that this resampled disk matches to!. 
    mesh_ref = unwrap3D_meshtools.create_mesh(np.hstack([np.zeros(len(disk_coords))[:,None], 
                                                            disk_coords]), mesh_2D_submesh.faces)
    disk_coords_relax_pts = disk_coords_relax[0].vertices.copy()
    
    
    uv_mesh_match = unwrap3D_meshtools.match_and_interpolate_uv_surface_to_mesh(np.hstack([np.zeros(len(disk_coords_relax_pts))[:,None], 
                                                                        disk_coords_relax_pts]).reshape(-1,3), 
                                                                        mesh_ref, match_method='cross')
    
    mesh_2D_submesh_pts = unwrap3D_meshtools.mesh_vertex_interpolate_scalar(mesh_ref, 
                                                                    uv_mesh_match[0], 
                                                                    uv_mesh_match[1], 
                                                                    mesh_2D_submesh.vertices)
    disk_coords = disk_coords_relax[0].vertices.copy()
    mesh_2D_disk = unwrap3D_meshtools.create_mesh(np.hstack([np.zeros(len(disk_coords))[:,None], 
                                          disk_coords]),
                                            disk_coords_relax[0].faces)
    
    mesh_2D_submesh = unwrap3D_meshtools.create_mesh(mesh_2D_submesh_pts,
                                                     disk_coords_relax[0].faces)
    
    if not conformal_map:
        # perform area relax. 
    
        # 5. Area relax the mesh 
        v_steps, f_steps, area_distortion_iter = area_distortion_flow_relax_disk(mesh_2D_disk, 
                                                                                 mesh_2D_submesh, 
                                                                                max_iter=areadistort_max_iter,
                                                                                # smooth_iters=5,  
                                                                                delta_h_bound=areadistort_delta_h_bound, 
                                                                                stepsize=areadistort_stepsize, 
                                                                                adaptive_step=False, # not yet implemented   
                                                                                flip_delaunay=area_distort_flip_tri, # breaking out because of this.... 
                                                                                robust_L=False, # use the robust laplacian instead of cotmatrix - slows flow. 
                                                                                mollify_factor=1e-5,
                                                                                eps = 1e-8,
                                                                                debugviz=False,
                                                                                lam=0, 
                                                                                debugviz_tri=debugviz_tri_areadistort)
        
        # 6. find the optimum point. 
        if len(area_distortion_iter) > 0:
            mean_area_distortion_iter = np.log(np.nanmean(np.array(area_distortion_iter),axis=1)) # we need to smooth this!
            
            # smooth this - as this is super bumpy. 
            mean_area_distortion_iter = baseline_als(mean_area_distortion_iter, lam=1000, p=0.1) # suppress the isolated bumps!!!!. 
            
            change = np.arange(len(mean_area_distortion_iter))[1:][np.diff(mean_area_distortion_iter)>0] 
            
            if len(change) > 0: 
                min_distort_id = change[0] - 5
            else:
                min_distort_id = len(mean_area_distortion_iter) - 5
            
            # construct the submesh!. 
            v_out = np.array(v_steps)[min_distort_id].copy()
            f_steps_out = np.array(f_steps)[min_distort_id].copy()  
            
        else:
            v_out = v_steps[0].copy()
            f_steps_out = f_steps[0].copy()
        
        # 7. Build the circular image 
        return v_out, f_steps_out, mesh_2D_submesh.vertices
    else:
        return disk_coords, mesh_2D_submesh.faces, mesh_2D_submesh.vertices

