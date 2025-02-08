import os
import numpy as np
from numba import njit
import shutil



def isolated_mask(indices, main_dict, main_list, min_pixel_dist=2):
    for elem in indices:
        frame = main_dict[main_list[elem][2]]
        index_mask = main_list[elem][3]
        masks = frame['masks']
        shape = masks.shape
        outline = frame['outlines'][index_mask - 1]
        if np.min(outline[:,0]) <= min_pixel_dist \
            or np.min(outline[:,1]) <= min_pixel_dist \
            or np.max(outline[:,0]) >= shape[0] -1 - min_pixel_dist \
            or np.max(outline[:,1]) >= shape[1] -1 - min_pixel_dist :
                return False
        if not check_isol(outline, masks, shape, index_mask, min_pixel_dist):
            return False
    return True

@njit
def check_isol(outline, masks, shape, index_mask, min_pixel_dist):
    for elem in outline:
        i,j = elem[0], elem[1]
        for k in range(-min_pixel_dist, min_pixel_dist+1):
            for l in range(-min_pixel_dist, min_pixel_dist+1):
                if 0 <= i+k < shape[0] and 0 <= j+l < shape[1]:
                    if masks[i+k, j+l] != 0 and masks[i+k, j+l] != index_mask:
                        return False
    return True


def main(dir_list):
    # dir_list=os.listdir('results')
    
    dicname='Main_dictionnary.npz'

    listname='masks_list.npz'

    ROIdict='ROI_dict.npz'

    saving_dir = 'cells'
    
    if os.path.exists(saving_dir):
        for folder in os.listdir(saving_dir):
            shutil.rmtree(os.path.join(saving_dir, folder))
    else:
        os.makedirs(saving_dir)
        
    count = 1 
    for dir in dir_list:
        path = os.path.join('results', dir)
        main_list=np.load(os.path.join(path, listname), allow_pickle=True)['arr_0']
        ROI_dict = np.load(os.path.join(path, ROIdict), allow_pickle=True)['arr_0'].item()
        main_dict=np.load(os.path.join(path, dicname), allow_pickle=True)['arr_0'].item()
        
        for ROI in ROI_dict:
            indices = ROI_dict[ROI]['Mask IDs']
            if len(indices)>=5 and ROI_dict[ROI]['Children']==[] and ROI_dict[ROI]['Parent']=='':
                good_masks = isolated_mask(indices, main_dict, main_list)
                if good_masks :
                    position = np.array([list(main_dict[main_list[id][2]]['repositionned_centroid'][main_list[id][3]-1]) for id in indices])
                    time = np.array([main_dict[main_list[id][2]]['time'] for id in indices])
                    area =  [int(main_dict[main_list[id][2]]['area'][main_list[id][3]-1]) for id in indices]
                    outlines = [main_dict[main_list[id][2]]['repositionned_outlines'][main_list[id][3]-1] for id in indices]
                    var_area = np.array([area[i+1]/area[i] for i in range(len(area)-1)])
                    area =  np.array(area)

                    if 0.75<=np.min(var_area) \
                        and  np.max(var_area)<=1.33 : #and np.max(np.abs(position[:-1]-position[1:]))<100 
                            cell_name = 'cell_'+str(count)
                            for frame_number in range(len(position)):
                                frame_name = 'frame_'+str(frame_number+1)
                                path = os.path.join(saving_dir, cell_name, frame_name)
                                if os.path.exists(path):
                                    for file in os.listdir(path):
                                        os.remove(os.path.join(path, file))
                                else:
                                    os.makedirs(path)
                                np.save(os.path.join(path, 'centroid.npy'),  position[frame_number]  )
                                np.save(os.path.join(path, 'time.npy'),   time[frame_number] )
                                np.save(os.path.join(path, 'outline.npy'),   outlines[frame_number] )
                            count+=1
                                    
                    
        
    
if __name__ == "__main__":
    os.chdir('..')
    dir_list = os.listdir('../data/')
    # dir_list = ['July6_plate1_xy02/', 'July6_plate1_xy05/', 'July6_plate1_xy06/',
    #             'July7_plate1_xy01/', 'July7_plate1_xy02/', 'July7_plate1_xy03/', 'July7_plate1_xy04/', 'July7_plate1_xy05/', 'July7_plate1_xy06/', 'July7_plate1_xy07/', 'July7_plate1_xy08/', 'July7_plate1_xy09/',
    #             'July8_plate1_xy01/', 'July8_plate1_xy02/', 'July8_plate1_xy04/',
    #             'July13_plate1_xy02 repositioned/', 'July13_plate1_xy05 repositioned/',  'July13_plate1_xy08/',  'July13_plate1_xy09/',  'July13_plate1_xy10/', 'July13_plate1_xy12/',
    #             'July15_plate1_xy01/', 'July15_plate1_xy02/', 'July15_plate1_xy03/']
    main(dir_list)
    
    
    

    
    # problem  'July13_plate1_xy03' time 7-8 'July13_plate1_xy07' time 7-8 'July13_plate1_xy11' time 7-8
    # problem 'July14_plate1_xy01' time 54 - 55  'July14_plate1_xy02' time 34 - 35  'July14_plate1_xy03' time 54 - 55 'July14_plate1_xy05' time 54 - 55