import os
from multiprocessing import Pool
import numpy as np
from copy import deepcopy


import processing.processing as pr
import processing.construct_dataset as cons_dat
import processing.final_graph as fg
import processing.plot_final_lineage_tree as pltfl
import motility_analysis.motility as mot


def reposition_centroids(dataset_time):
    dataset, time = dataset_time[0], dataset_time[1]
    path = os.path.join('results', dataset)
    dicname = os.path.join(path,'Main_dictionnary.npz')
    main_dict = np.load(dicname, allow_pickle=True)['arr_0'].item()
    dic_list = list(main_dict.keys())
    previous_img = pr.main_mask(main_dict[dic_list[time-1]]['masks'])
    next_img = pr.main_mask(main_dict[dic_list[time]]['masks'])
    vec = pr.opt_trans_vec2(previous_img, next_img)
    
    for i in range(time, len(dic_list)):
        main_dict[dic_list[i]]['repositionned_centroid'] += vec
        for val in main_dict[dic_list[i]]['repositionned_outlines']:
            val += vec
    
    np.savez_compressed(dicname, main_dict, allow_pickle=True)
         

  

def main(direc):
    pr.run_end_preprocess_watershed(direc)
    fg.Final_lineage_tree(direc)    
    pltfl.run_whole_lineage_tree(direc, show=False)


if __name__ == "__main__":

    reposition_list = [['July13_plate1_xy03', 7], ['July13_plate1_xy07', 7], ['July13_plate1_xy08', 7],  ['July13_plate1_xy09', 7], ['July13_plate1_xy11', 7], ['July14_plate1_xy01', 43],  ['July14_plate1_xy02', 34],  ['July14_plate1_xy03', 43], ['July14_plate1_xy05', 43]]

    
    dir_list = os.listdir('../data/')
    for direc in dir_list:
        pr.run_cellpose(direc)
    
    with Pool(processes=8) as pool:
            for direc in pool.imap_unordered(main, dir_list):
                pass
    
    
    with Pool(processes=8) as pool:
            for direc in pool.imap_unordered(reposition_centroids,reposition_list):
                pass
            
    
    cons_dat.main(dir_list)
    mot.main()