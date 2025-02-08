import os 
import numpy as np 
import matplotlib.pyplot as plt

def conformal_representation(cell_folder ,
                             ref_disk_radius = 50,
                             ref_iter_no = 3, 
                             lambda_flow = 200, 
                             curvature_threshold = 0.1):
    
    import glob
    import skimage.io as skio 
    import pylab as plt 
    import skimage.transform as sktform 
    
    import unwrap2D.unwrap2D as unwrap2D_fns
    import unwrap3D.Mesh.meshtools as unwrap3D_meshtools
    import unwrap3D.Segmentation.segmentation as unwrap3D_segmentation
    
    
    # https://github.com/clementsoubrier/shape_analysis/blob/main/construct_dataset.py


    # ref_disk_radius = 50
    # ref_iter_no = 3 # 3 use this and 
    # lambda_flow = 200 # 5e1  use this to tune the extent of smoothing of base shape, the less the smoother 
    # curvature_threshold = 0.1 # only apply flow iterations if np.mean(np.abs(curvature)) > threshold i.e. there are potential protrusions! 
    
    
    """
    Iterate over cells. 
    """


    times = np.sort(os.listdir(cell_folder)) # perform nature sort
    times_int = np.hstack([int(tt.split('frame_')[1]) for tt in times])
    natsort = np.argsort(times_int)
    
    times_int = times_int[natsort]
    times = times[natsort]
    
    all_outlines = []
    all_outlines_cMCF = []
    all_outlines_cMCF_topography = []
    all_outlines_curvature = []
    fin_times = []
    fin_centr = []
    
    for ttt in np.arange(len(times))[:]:
        
        outline_ttt_file = os.path.join(cell_folder, times[ttt], 'outline.npy')
        centroid_ttt_file = os.path.join(cell_folder, times[ttt], 'centroid.npy')
        time_ttt_file = os.path.join(cell_folder, times[ttt], 'time.npy')
        
        outline = np.load(outline_ttt_file)
        all_outlines.append(outline)
        fin_times.append(np.load(time_ttt_file))
        centroid = np.load(centroid_ttt_file)
        fin_centr.append(centroid)
        # evolve the contour. 
        outline_input = outline.copy()
        _,_, outline_input_curvature = unwrap2D_fns.curvature_splines(outline_input[:,1], 
                                                                outline_input[:,0],
                                                                k=4,
                                                                error=0.1)
        all_outlines_curvature.append(outline_input_curvature)
        outline_input_curvature_norm = np.nanmean(np.linalg.norm(outline_input-centroid[None,:], axis=-1), axis=0)/2. * outline_input_curvature

        # print(outline_input_curvature_norm.min(),outline_input_curvature_norm.max(), np.nanmean(np.abs(outline_input_curvature_norm)))
                    
        # =============================================================================
        #            Find reference
        # =============================================================================
        contour_evolve = unwrap3D_meshtools.conformalized_mean_line_flow( np.hstack([np.ones(len(outline_input))[:,None], 
                                                                    outline_input]), 
                                                            E=None, 
                                                            close_contour=True, 
                                                            fixed_boundary = False, 
                                                            lambda_flow=lambda_flow, # change this to control the degree of smoothing... 
                                                            niters=20, # hm... increasing this helps find ref.  
                                                            topography_edge_fix=False, 
                                                            conformalize=True)
        contour_evolve = contour_evolve[:,1:]
        '''
        # mask_contour = unwrap2D_fns.pixelize_contours(outline, 
        #                                               imshape=None,
        #                                               padsize=50)
        # H_normal, sdf_vol_normal, sdf_vol = unwrap3D_segmentation.mean_curvature_binary(mask_contour, 
        #                                                                         smooth=1, 
        #                                                                         mask=False, 
        #                                                                         smooth_gradient=3, eps=1e-12)
        
        # contour_evolve = unwrap2D_fns.parametric_line_flow_2D(outline_input,
        #                                                         external_img_gradient=sdf_vol_normal.transpose(1,2,0), 
        #                                                         E=None, 
        #                                                         close_contour=True, 
        #                                                         fixed_boundary = False, 
        #                                                         lambda_flow=5e1, # adjusts the balance.  (decrease this to be more similar to original cell shape ), increase to be more like version 1
        #                                                         step_size=0.25, # adjusts the spacing between curves. 
        #                                                         niters=20, 
        #                                                         conformalize=False,
        #                                                         eps=1e-12)
        '''
        # register the contours to original 
        contour_evolve_register = unwrap2D_fns.register_contour_array(contour_evolve, ref_id=0, tform='Similarity')
        
        # errs = np.linalg.norm(contour_evolve-contour_evolve[...,0][...,None], axis=1).mean(axis=0)
        errs_register = np.linalg.norm(contour_evolve_register-contour_evolve_register[...,0][...,None], axis=1).mean(axis=0)
        '''
        # errs_register = np.nanmean(np.cumsum(np.linalg.norm(np.diff(contour_evolve, axis=-1), axis=1), axis=-1), axis=0)
        
        # inds = unwrap3D_meshtools.find_all_curvature_cutoff_index(errs_register, winsize=5,
        #                                                           min_peak_height=0)
        # # inds_original = unwrap3D_meshtools.find_all_curvature_cutoff_index(errs, winsize=5)
        # if len(inds) == 0:
        #     ind = 0
        # else:
        #     ind = inds[0] + 1 
        # # ind = inds_original[0] #+ 1 
        '''
        if np.nanmean(np.abs(outline_input_curvature_norm)) > curvature_threshold:
            ind = ref_iter_no
        else:
            ind = 1 
        
        # plt.figure(figsize=(10,10))
        # plt.plot(outline[:,1], 
        #           outline[:,0], 'k-', lw=10)
        # plt.plot(contour_evolve_register[:,1,ind], 
        #             contour_evolve_register[:,0,ind], 'g-', lw=10)
        # plt.show()
    
    
        """
        map to circle 
        """
        mask_contour = unwrap2D_fns.pixelize_contours(outline, 
                                                    imshape=None,
                                                    padsize=50)
        
        contour_ref = contour_evolve_register[...,ind].copy()
        contour_ref_protrusion_d, sign_d = unwrap2D_fns.measure_signed_distance(contour_ref, 
                                                                        contours=contour_evolve_register.transpose(2,0,1)[:ind+1], 
                                                                        imshape=mask_contour.shape)
        
        all_outlines_cMCF.append(contour_ref)
        
        topography_coords_disk, disk_coords = unwrap2D_fns.map_protrusion_topography_to_circle(contour_ref, 
                                                                                        contour_ref_protrusion_d, 
                                                                                        circle_R=ref_disk_radius, 
                                                                                        close_contour=True)
    
        
        all_outlines_cMCF_topography.append(topography_coords_disk)
        
        
    return all_outlines, all_outlines_cMCF, all_outlines_cMCF_topography, all_outlines_curvature, disk_coords, np.array(fin_centr), np.array(fin_times)






def max_protusion_plot(direct):
    
    all_cell_folders = [os.path.join(direct, ff) for ff in os.listdir(direct)]
    for cell_folder_ii in np.arange(len(all_cell_folders)):
        all_outlines, _, all_outlines_cMCF_topography, all_outlines_curvature, disk_coords, _, fin_times = conformal_representation(all_cell_folders[cell_folder_ii])
     
        plt.figure()
        plt.title(f'Shape evolution, cell {cell_folder_ii}')
        ax = plt.axes(projection='3d')
        tot_len = len(all_outlines_cMCF_topography) #10
        colors = plt.cm.viridis(np.linspace(0,1,tot_len))
        max_top = []
        max_curv = []

        for jj in np.arange(tot_len) :
            topography_coords_disk = all_outlines_cMCF_topography[jj]
            outline = all_outlines[jj]
            time = fin_times[jj]
            curv = all_outlines_curvature[jj]
            index = np.argmax(topography_coords_disk [:,0]**2+topography_coords_disk [:,1]**2)
            index_curvature = np.argmin(curv)

            ax.plot3D(outline[:,0],outline[:,1], time*np.ones(len(outline[:,1])), c=colors[jj] )
            max_top.append([outline[index][0], outline[index][1], time])
            max_curv.append([outline[index_curvature][0], outline[index_curvature][1], time])
            
        max_top = np.array(max_top)
        max_curv = np.array(max_curv)   
        
        # ax.scatter(max_top[:,0], 
        #                 max_top[:,1], max_top[:,2], c='k', label='max topo height')
        ax.scatter(max_curv[:,0], 
                        max_curv[:,1], max_curv[:,2], c='r', label='max curvature')
        ax.legend()
        
            
        plt.figure()
        plt.title(f'Topographical representation, cell {cell_folder_ii}')
        ax = plt.axes(projection='3d')
        tot_len = len(all_outlines_cMCF_topography) #10
        colors = plt.cm.viridis(np.linspace(0,1,tot_len))
        max_top = []
        max_curv = []
        
        for jj in np.arange(tot_len) :
            topography_coords_disk = all_outlines_cMCF_topography[jj]
            outline = all_outlines[jj]
            time = fin_times[jj]
            curv = all_outlines_curvature[jj]
            index = np.argmax(topography_coords_disk [:,0]**2+topography_coords_disk [:,1]**2)
            index_curvature = np.argmin(curv)

            ax.plot3D(topography_coords_disk[:,0],topography_coords_disk[:,1], time*np.ones(len(outline[:,1])), c=colors[jj] )
            max_top.append([topography_coords_disk[index][0], topography_coords_disk[index][1], time])
            max_curv.append([topography_coords_disk[index_curvature][0], topography_coords_disk[index_curvature][1], time])
            
        max_top = np.array(max_top)
        max_curv = np.array(max_curv)  
         
        ax.plot3D(disk_coords[:,0], disk_coords[:,1], np.zeros(len(disk_coords[:,0])), c='k')
        ax.plot3D(disk_coords[:,0], disk_coords[:,1], time*np.ones(len(disk_coords[:,0])), c='k')
        # ax.scatter(max_top[:,0], 
        #                 max_top[:,1], max_top[:,2], c='k', label='max topo height')
        ax.scatter(max_curv[:,0], 
                        max_curv[:,1], max_curv[:,2], c='r', label='max curvature')
        ax.legend()
        plt.show()
            
def _stats_max_protusions(cell_folder, smoothing = True, smooth_time = 4, smooth_space=2):
    all_outlines, _, all_outlines_cMCF_topography, all_outlines_curvature, _, fin_centr, fin_times = conformal_representation(cell_folder)
    tot_len = len(all_outlines)
    vec_topo = []
    vec_curv = []
    val_curv = []
    for jj in np.arange(tot_len)[1:]:
        
        topography_coords_disk = all_outlines_cMCF_topography[jj]
        outline = all_outlines[jj]
        curv = all_outlines_curvature[jj]
        centr = fin_centr[jj]
        
        index = np.argmax(topography_coords_disk [:,0]**2+topography_coords_disk [:,1]**2)
        index_curv = np.argmin(curv)
        
        vec_topo.append(outline[index] - centr)
        vec_curv.append(outline[index_curv] - centr)
        range_m = [max(0,index_curv-smooth_space),min(len(curv)-1,index_curv+smooth_space)]
        val_curv.append(np.median(curv[range_m[0]:range_m[1]]))
        
    displacement = fin_centr[1:]- fin_centr[:-1]
    velocity = np.zeros(np.shape(displacement))
    new_curv = val_curv =  np.array(val_curv)
    if smoothing:
        
        for jj, val_jj in enumerate(displacement):
            mask = np.logical_and(fin_times[1:]>= fin_times[jj] - smooth_time, fin_times[1:]<= fin_times[jj] + smooth_time)
            if np.sum(mask) >1 :
                time_var = fin_times[1:] - fin_times[:-1]
                velocity[jj] = (np.sum(displacement[mask])) / (np.sum(time_var[mask]))
                val_curv[jj] = np.mean(new_curv[mask])
            else:
                velocity[jj] = val_jj / (fin_times[jj+1] - fin_times[jj])
            
    else :
        velocity[:,0] = displacement[:,0] / (fin_times[1:]-fin_times[:-1])
        velocity[:,1] = displacement[:,1] / (fin_times[1:]-fin_times[:-1])
        

    return velocity, vec_topo, vec_curv, list(val_curv)

def stats_max_protusions(direct):
    from multiprocessing import Pool
    res_velo = []
    res_topo = [] 
    res_curv = []
    val_curv = []
    all_cell_folders = [os.path.join(direct, ff) for ff in os.listdir(direct)]
    
    
    with Pool(processes=8) as pool:
            for _velo, _topo, _curv, _vcurv in pool.imap_unordered(_stats_max_protusions, all_cell_folders):
                val_curv.extend(_vcurv)
                res_velo.extend(_velo)
                res_topo.extend(_topo)
                res_curv.extend(_curv)
            
    res_curv = np.array(res_curv)
    res_velo = np.array(res_velo)
    res_topo = np.array(res_topo)
    val_curv = np.array(val_curv)
    
    plt.figure()
    bin_num = 20
    plt.xlabel('Max curvature')
    plt.ylabel('Velocity norm')
    vel_norm = (res_velo[:,0]**2+res_velo[:,1]**2)**0.5
    
    hist = np.linspace(np.min(-val_curv), np.max(-val_curv)+0.001,bin_num+1)
    res = np.zeros(bin_num)
    for ii in range(bin_num):
        mask = np.logical_and(val_curv<hist[ii+1], -val_curv>=hist[ii])
        res[ii] = np.mean(vel_norm[mask])
    print(hist, res)
    plt.plot(hist[1:], res)
    print('velocity stats : ',np.mean(vel_norm), np.median(vel_norm), np.std(vel_norm))
    # plt.scatter(-val_curv, vel_norm)
    
    
    
    plt.figure()
    bin_num = 20
    bottom = 400
    theta_topo = np.arctan2(res_topo[:,0], res_topo[:,1])
    theta_velo = np.arctan2(res_velo[:,0], res_velo[:,1])
    theta = (theta_topo - theta_velo ) % (2 * np.pi)
    
    hist = np.histogram(theta, bins=bin_num, range=(0,2 * np.pi))
    width = (2*np.pi) / bin_num

    ax = plt.subplot(111, polar=True)
    ax.set_title('Angle of max protusion with the displacement, topology')
    bars = ax.bar((hist[1][:-1]+hist[1][1:])/2, hist[0], width=width, bottom=bottom)

    # Use custom colors and opacity
    max_val = np.max(hist[0])
    for r, bar in zip(hist[0], bars):
        bar.set_facecolor(plt.cm.jet(r / max_val))
        bar.set_alpha(0.8)
        
    
    plt.figure()
    bin_num = 20
    bottom = 400

    mask_velo = vel_norm >=  np.median(vel_norm)*1.5
    theta_curv = np.arctan2(res_curv[:,0][mask_velo], res_curv[:,1][mask_velo])
    theta_velo = np.arctan2(res_velo[:,0][mask_velo], res_velo[:,1][mask_velo])
    theta = (theta_curv - theta_velo) % (2 * np.pi)
    
    hist = np.histogram(theta, bins=bin_num, range=(0,2 * np.pi))
    width = (2*np.pi) / bin_num

    ax = plt.subplot(111, polar=True)
    ax.set_title('Angle of max protusion with the displacement, high speed, curvature')
    bars = ax.bar((hist[1][:-1]+hist[1][1:])/2, hist[0], width=width, bottom=bottom)

    # Use custom colors and opacity
    max_val = np.max(hist[0])
    for r, bar in zip(hist[0], bars):
        bar.set_facecolor(plt.cm.jet(r / max_val))
        bar.set_alpha(0.8)
        
    plt.figure()
    bin_num = 20
    bottom = 400

    non_mask_velo = np.logical_not(mask_velo)
    theta_curv = np.arctan2(res_curv[:,0][non_mask_velo], res_curv[:,1][non_mask_velo])
    theta_velo = np.arctan2(res_velo[:,0][non_mask_velo], res_velo[:,1][non_mask_velo])
    theta = (theta_curv - theta_velo) % (2 * np.pi)
    
    hist = np.histogram(theta, bins=bin_num, range=(0,2 * np.pi))
    width = (2*np.pi) / bin_num

    ax = plt.subplot(111, polar=True)
    ax.set_title('Angle of max protusion with the displacement, low speed, curvature')
    bars = ax.bar((hist[1][:-1]+hist[1][1:])/2, hist[0], width=width, bottom=bottom)

    # Use custom colors and opacity
    max_val = np.max(hist[0])
    for r, bar in zip(hist[0], bars):
        bar.set_facecolor(plt.cm.jet(r / max_val))
        bar.set_alpha(0.8)

    
    plt.show()




if __name__ == '__main__':
    glob_folder = 'cells'
    
    max_protusion_plot(glob_folder)
    # stats_max_protusions(glob_folder)
    
    

    
        
