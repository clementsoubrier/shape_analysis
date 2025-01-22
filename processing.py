#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:18:29 2022

@author: c.soubrier
"""

#First test AFM

import os
import cv2
from copy import deepcopy
import numpy as np
from numba import njit
import tqdm
from cellpose import utils, io, models, denoise
from skimage.morphology import skeletonize, thin
from skimage.segmentation import watershed
import re
from shutil import rmtree #erasing a whole directory
import scipy
from PIL import Image
import matplotlib.pyplot as plt





''' Paths of the data and to save the results'''


# =============================================================================
# #Inputs
# #directory of the original dataset composed of a sequence of following pictures of the same bacterias, and with log files with .001 or no extension
# Main_directory=   #the directory you chos to work on
# dir_name=Main_directory+"Height/"             #name of dir
# my_data = "../data/"+dir_name       #path of dir
# #directory with the usefull information of the logs, None if there is no.
# data_log="../data/"+Main_directory+"log/"             #       #path and name
# 
# #Temporary directories
# #directory of cropped data each image of the dataset is cropped to erase the white frame
# cropped_data=dir_name+"cropped_data/" 
# #directory with the usefull info extracted from the logs
# cropped_log=dir_name+"log_cropped_data/" 
# #directory for output data from cellpose 
# segments_path = dir_name+"cellpose_output/"
# 
# 
# #Outputs
# #directory of the processed images (every image has the same pixel size and same zoom)
# final_data=dir_name+"final_data/" 
# #dictionnary and dimension directory
# Dic_dir=dir_name+"Dic_dir/" 
# #Saving path for the dictionnary
# saving_path=Dic_dir+'Main_dictionnary'
# #dimension and scale of all the final data
# dimension_data=Dic_dir+'Dimension'
# =============================================================================

# different type of datassets with their cropping parameter










''' main dictionnary main_dict: each image contain a dictionnary with
time : the time of the image beginning at 0
adress : location of the cropped file
masks : the masks as numpy array
resolution : physical resolution of a pixel
outlines : the outlines as a list of points
angle: rotation angle of the image compared to a fixed reference
centroid : an array containing the position of the centroid of each mask the centroid[i] is the centroid of the mask caracterized by i+1
area : an array containing the position of the area (number of pixels) of each mask
mask_error : True if cellpose computes no mask, else False
# convex_hull : convex hull of each mask
main_centroid : the centroid of all masks
parent / child : previous / next acceptable  file
mask_list : index of the mask in the masks
'''

def natural_keys(text):
    '''
    tool function to sort files by timepoints

    Parameters
    ----------
    text : string
        
        

    Returns
    -------
    list
        list of the digits

    '''
    l = [ int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text) ]
    l.reverse()
    return l



@njit    
def renorm_img(img):
    dim1,dim2=np.shape(img)
    newimg=np.zeros((dim1,dim2))
    maxi=np.max(img)
    mini=np.min(img)
    if maxi==mini:
        return newimg
    else:
        for i in range(dim1):
            for j in range(dim2):
                newimg[i,j]=(img[i,j]-mini)/(maxi-mini)
        return newimg
    
    
    
#%%
def run_cell_simple(data,mod,chan,dia,thres,celp,seg,denoise_mod,gpuval=True):
    if os.path.exists(seg):
        for file in os.listdir(seg):
            os.remove(os.path.join(seg, file))
    else:
        os.makedirs(seg)
    listdir=os.listdir(data)
    for i in tqdm.trange(len(listdir)):
        img=np.array(Image.open(os.path.join(data, listdir[i])))[:,:,1]
        if gpuval:
            try:
                # Specify that the cytoplasm Cellpose model for the segmentation. 
                model = denoise.CellposeDenoiseModel(gpu=True, model_type=mod, restore_type=denoise_mod)
                masks, flows, st, diams = model.eval(img, diameter = dia, channels=chan, flow_threshold = thres, cellprob_threshold=celp)
            except:
                model = denoise.CellposeDenoiseModel(gpu=False, model_type=mod, restore_type=denoise_mod)
                masks, flows, st, diams = model.eval(img, diameter = dia, channels=chan, flow_threshold = thres, cellprob_threshold=celp)
        else :
            model = denoise.CellposeDenoiseModel(gpu=False, model_type=mod, restore_type=denoise_mod)
            masks, flows, st, diams = model.eval(img, diameter = dia, channels=chan, flow_threshold = thres, cellprob_threshold=celp)
        io.masks_flows_to_seg(img, masks, flows, os.path.join(seg+listdir[i][:-4]), channels=chan, diams=diams)
        
        
def run_cell_boundary(data,mod,chan,dia,flow_small,flow_big,celp_small,celp_big,seg,surfcomthres,boundarythres,denoise_mod, gpuval=True):
    if os.path.exists(seg):
        for file in os.listdir(seg):
            os.remove(os.path.join(seg, file))
    else:
        os.makedirs(seg)
    listdir=os.listdir(data)
    
    
    for i in tqdm.trange(len(listdir)):
        img=np.array(Image.open(data+listdir[i]))[:,:,1]
        
        
        if gpuval:
            try:
                # Specify that the cytoplasm Cellpose model for the segmentation. 
                model = denoise.CellposeDenoiseModel(gpu=True, model_type=mod, restore_type=denoise_mod)
                masks1, flows1, st1, diams1 = model.eval(img, diameter = dia, channels=chan, flow_threshold = flow_small, cellprob_threshold=celp_small)
            except:
                model = denoise.CellposeDenoiseModel(gpu=False, model_type=mod, restore_type=denoise_mod)
                masks1, flows1, st1, diams1 = model.eval(img, diameter = dia, channels=chan, flow_threshold = flow_small, cellprob_threshold=celp_small)
        else :
            model = denoise.CellposeDenoiseModel(gpu=False, model_type=mod, restore_type=denoise_mod)
            masks1, flows1, st1, diams1 = model.eval(img, diameter = dia, channels=chan, flow_threshold = flow_small, cellprob_threshold=celp_small)
            
            
        if gpuval:
            try:
                # Specify that the cytoplasm Cellpose model for the segmentation. 
                model = denoise.CellposeDenoiseModel(gpu=True, model_type=mod, restore_type=denoise_mod)
                masks2, flows2, st2, diams2 = model.eval(img, diameter = dia, channels=chan, flow_threshold = flow_big, cellprob_threshold=celp_big)
            except:
                model = denoise.CellposeDenoiseModel(gpu=False, model_type=mod, restore_type=denoise_mod)
                masks2, flows2, st2, diams2 = model.eval(img, diameter = dia, channels=chan, flow_threshold = flow_big, cellprob_threshold=celp_big)
        else :
            model = denoise.CellposeDenoiseModel(gpu=False, model_type=mod, restore_type=denoise_mod)
            masks2, flows2, st2, diams2 = model.eval(img, diameter = dia, channels=chan, flow_threshold = flow_big, cellprob_threshold=celp_big)
        
        masking_masks=masks2==0
        low_val=np.median(img[np.logical_not(masking_masks)])
        newmask=np.zeros(np.shape(masks2),dtype=np.uint8)
        
        newindex=1
        for index in range(1,np.max(masks1)+1):
            surf=np.count_nonzero(masks1==index)
            poss_index=np.zeros(np.max(masks1))
            for j in range (np.max(masks1)):
                poss_index[j]=np.count_nonzero(np.logical_and(masks1==index,masks2==j+1))
            link=np.argmax(poss_index)
            if poss_index[link]>=surf*surfcomthres:
                val= low_val+(np.average(img[masks1])-low_val)*boundarythres
                
                bool_frame=np.logical_or(masks1==index,np.logical_and(masks2==link+1,img>=val))
                if np.max(bool_frame):
                    contour=cv2.findContours(bool_frame.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]#[0][:,0,:]
                   
                    if len(contour)==1:
                        cv2.drawContours(newmask,[contour[0][:,0,:]],-1,color=newindex, thickness=cv2.FILLED)
                    else:
                        
                        k=np.argmax(np.array([len(contour[l]) for l in range(len(contour))]))
                        cv2.drawContours(newmask,[contour[k][:,0,:]],-1,color=newindex, thickness=cv2.FILLED)
                        
                    newindex+=1
        unique=np.unique(newmask)
        uniquelen=len(np.unique(newmask))
        for j in range(1, uniquelen):
            newmask[newmask==unique[j]]=j
        io.masks_flows_to_seg(img, newmask, flows1, diams1, seg+listdir[i][:-4], chan)

#%%
def download_dict_logs_only(mydata,dir,segmentspath,saving=True,savingpath='dict'):
    
    dir_im = os.path.join(mydata,dir)
    files = os.listdir(dir_im)
    dic={}
    
    t=0
    # Sort files by timepoint.
    files.sort(key = natural_keys)
    for fichier in files:
        fichier=fichier[:-4]
        dat = np.load(segmentspath+fichier+'_seg.npy', allow_pickle=True).item()
        if np.max(dat['masks'])!=0:

            dic[fichier]={}
            
            
            
            dic[fichier]['time']=t      #time in minutes after the beginning of the experiment
            dic[fichier]['adress']= os.path.join(dir_im, fichier+'.tif')
            dic[fichier]['masks']=dat['masks']
            dic[fichier]['outlines']=utils.outlines_list(dat['masks'], multiprocessing=False)
            dic[fichier]['angle']=0
        t+=1
    #deleting temporary dir
    rmtree(dir)
    if saving:
        np.savez_compressed(savingpath,dic)
    return dic


    

#%% Construction the sequence of usable pictures : linking images with previous (parent) and following (child) image
def main_parenting(dic,saving=False,savingpath='dict'):
    parent=''
    list_key=list(dic.keys())
    for fichier in list_key:
            dic[fichier]['parent']=parent
            parent=fichier
    child=''
    key=list(dic.keys())
    key.reverse()
    for fichier in key:
            dic[fichier]['child']=child
            child=fichier
    if saving:
        np.savez_compressed(savingpath,dic)



#%% Computing the area and the centroid of each mask and saving them in the dictionnary
def centroid_area(dic):
    diclist=list(dic.keys())
    for i in tqdm.trange(len(diclist)):
        fichier=diclist[i]
        masks=dic[fichier]['masks']
        (centroid,area)=numba_centroid_area(masks)
        dic[fichier]['centroid']=centroid
        dic[fichier]['area']=area

#to improve computation speed
@njit
def numba_centroid_area(masks):
    mask_number=np.max(masks)
    centroid=np.zeros((mask_number,2),dtype=np.int32)
    area=np.zeros(mask_number,dtype=np.int32)
    (l,L)=np.shape(masks)
    for i in range(mask_number):
        count=0
        vec1=0
        vec2=0
        for j in range(L):
            for k in range(l):
                if masks[k,j]==i+1:
                    vec1+=j
                    vec2+=k
                    count+=1
        area[i]=count
        
        centroid[i,:]=np.array([vec2//count,vec1//count],dtype=np.int32)
    return(centroid,area)

#watershed sub routine
def _watershed(masks,tresh):
    marker = np.zeros(masks.shape,dtype=int)
    for i in range(1, np.max(masks)+1):
        submask = masks == i
        submask = submask.astype(np.uint8)
        dist_transform = cv2.distanceTransform(submask,cv2.DIST_L2,5)
        _, sure_fg = cv2.threshold(dist_transform,tresh*dist_transform.max(),255,0) 
        non_zero = sure_fg > 0
        marker[non_zero] = i
            
    return marker.astype(np.int32)


# re-segmenting unsing the watershed algorithm
def mask_watershed(dic,saving=False,savingpath='dict'):
    diclist=list(dic.keys())
    for i in tqdm.trange(len(diclist)):
        fichier = diclist[i]
        img = cv2.imread(dic[fichier]['adress']).astype(np.uint8) 
        
        masks = np.array(dic[fichier]['masks'])
        max_mask = np.max(masks)
        max_surf = np.max(np.array([np.sum(masks==i) for i in range (1, max_mask+1)]))
        markers = _watershed(masks,0.2)
        markers2 = _watershed(masks,0.1)
        new_masks = cv2.watershed(img,np.array(markers))
        
        for i in range (1, max_mask+1):
            if np.sum(new_masks==i)> 1.5*max_surf:
                new_masks[new_masks==i] = 0
                new_masks[markers2==i] = i
        new_masks[new_masks<1] = 0
        
        img = np.array(Image.open(dic[fichier]['adress']))[:,:,1]
        new_new_masks = watershed(img,markers=np.array(new_masks),compactness=0)
        for i in range (1, max_mask+1):
            if np.sum(new_new_masks==i)> 1.5*max_surf:
                new_new_masks[new_new_masks==i] = 0
                new_new_masks[new_masks==i] = i
        
        
        
        dic[fichier]['masks'] = new_new_masks
        dic[fichier]['outlines']=utils.outlines_list(new_new_masks, multiprocessing=False)
    if saving:
        np.savez_compressed(savingpath,dic)


#%% Erasing too small masks (less than the fraction frac_mask of the largest mask), and mask with a ratio of saturated area superior to frac_satur . Creating the centroid of the union of acceptable  mask and saving as main_centroid
def clean_masks(frac_mask,dic,saving=False,savingpath='dict'): 
    #Erase the masks that are too small (and the centroids too)
    diclist=list(dic.keys())
    for j in tqdm.trange(len(diclist)):
        fichier=diclist[j]
        masks=dic[fichier]['masks']
        area=dic[fichier]['area']
        centroid=dic[fichier]['centroid']
        outlines=dic[fichier]['outlines']
        
        
        max_area=np.max(area)
        L=len(area)
        non_defect=np.zeros(L) #classification of the allowed masks
        non_defect_count=0
        newoutlines=[]
        
        
        for i in range(L):
            if area[i]>=frac_mask*max_area:
                non_defect_count+=1
                non_defect[i]=non_defect_count
                newoutlines.append(outlines[i][:,::-1].astype(np.int32))
                
        #update the outlines
        dic[fichier]['outlines']=newoutlines
        dic[fichier]['repositionned_outlines']=deepcopy(newoutlines)
        #new value of the area and the centroid
        area2=np.zeros(non_defect_count).astype(np.int32)
        centroid2=np.zeros((non_defect_count,2),dtype=np.int32)
        for i in range(L):
            if non_defect[i]!=0:
                area2[int(non_defect[i]-1)]=area[i]
                centroid2[int(non_defect[i]-1),:]=centroid[i,:]
        (m,n)=masks.shape
        for i in range(m):
            for k in range(n):
                if masks[i,k]!=0:
                    masks[i,k]=non_defect[masks[i,k]-1]
        
        dic[fichier]['area']=area2
        dic[fichier]['centroid']=centroid2
        dic[fichier]['repositionned_centroid']=np.array(centroid2)
        #constructing the main centroid
        if sum(area2)>0:
            main_centroid0=0
            main_centroid1=0
            for i in range (non_defect_count):
                main_centroid0+=area2[i]*centroid2[i,0]
                main_centroid1+=area2[i]*centroid2[i,1]
            
            dic[fichier]['main_centroid']=np.array([main_centroid0//sum(area2),main_centroid1//sum(area2)],dtype=np.int32)
        else :
            del dic[fichier]
    if saving:
        np.savez_compressed(savingpath,dic)



 
# creating a new mask with changed values
@njit
def update_masks(mask,new_values):
    (l,L)=np.shape(mask)
    for j in range(l):
        for k in range(L):
            if mask[j,k]!=0:
                mask[j,k]=new_values[mask[j,k]-1]
    return mask

#%%Contructing a list containing all the masks of the dataset with this structure : list_index,dataset, frame,mask_index

def construction_mask_list(dic,listsavingpath,saving=False,savingpath='dict'):
   index=0
   final_list=[]
   diclist=list(dic.keys())
   for i in tqdm.trange(len(diclist)):
        fichier=diclist[i]
        masks=dic[fichier]['masks']
        mask_number=np.max(masks)
        list_index=[]
        for i in range(mask_number):
            final_list.append([index,'',fichier,i+1])
            list_index.append(index)
            index+=1
        dic[fichier]["mask_list"]=list_index
   np.savez_compressed(listsavingpath,np.array(final_list,dtype=object))
   if saving:
       np.savez_compressed(savingpath,dic)




@njit
def transfo_bin(mask,num):
    (dim1,dim2)=np.shape(mask)
    newmask=np.zeros((dim1,dim2),dtype=np.int32)
    for i in range(dim1):
        for j in range(dim2):
            if mask[i,j]==num:
                newmask[i,j]=1
    return newmask


''' Computing the translation vector between an image and its child and saving it under translation_vector'''

# Tranform all masks into one shape (the main shape)
@njit
def main_mask(mask):
    (l,L)=np.shape(mask)
    new_mask=np.zeros((l,L))
    for i in range(l):
        for j in range(L):
            if mask[i,j]!=0:
                new_mask[i,j]=1
    return new_mask

# Define the score of a function (here a sum of white pixels)
@njit 
def score_mask(mask1,mask2):
    (l1,l2)=np.shape(mask1)
    score=0
    for i in range(l1):
        for j in range(l2):
            if mask1[i,j]==1 and mask2[i,j]==1:
                score+=1
    return score
         
# Translation of the masks by a vector
@njit 
def mask_transfert(mask,vector):
    (l1,l2)=np.shape(mask)
    new_mask=np.zeros((l1,l2),dtype=np.int32)
    for i in range(l1):
        for j in range(l2):
            if (0<=i+vector[0]<=l1-1) and (0<=j+vector[1]<=l2-1) and mask[i,j]>0:
                new_mask[int(i+vector[0]),int(j+vector[1])]=mask[i,j]
    return new_mask
      
# Effective computation of the translation vector : returns the translation vector that optimizes the score of the intersection of the two main shapes

def opt_trans_vec2(img_1, img_2):
    corr = scipy.signal.fftconvolve(img_1, img_2[::-1, ::-1])
    argmax = np.unravel_index(corr.argmax(), corr.shape)
    vec = np.array(argmax) - np.array(img_1.shape) + 1
    return vec

#rotation of a vector around a point, the angle is in radian (float as output)
@njit 
def rotation_vector(angle,vec,point):
    mat=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    newvec=vec-point
    var=np.array([mat[0,0]*newvec[0]+mat[0,1]*newvec[1],mat[1,0]*newvec[0]+mat[1,1]*newvec[1]])
    return point+var

#rotation of an image around a point (int32 as input)
@njit 
def rotation_img(angle,img,point):
    dim1,dim2=np.shape(img)
    new_img=np.zeros((dim1,dim2),dtype=np.int32)
    for i in range(dim1):
        for j in range(dim2):
            trans_vec=rotation_vector(-angle,np.array([i,j]),point)#sign of the rotation : definition of the angle in the logs
            i_n,j_n=trans_vec[0],trans_vec[1]
            i_t=int(i_n)
            j_t=int(j_n)
            if 0<=i_t<dim1-1 and 0<=j_t<dim2-1:
                frac_i=i_n-i_t
                frac_j=j_n-j_t
                new_img[i,j]=np.int32(frac_i*frac_j*img[i_t,j_t]+frac_i*(1-frac_j)*img[i_t,j_t+1]+(1-frac_i)*frac_j*img[i_t+1,j_t]+(1-frac_i)*(1-frac_j)*img[i_t+1,j_t+1])
    return new_img




def skeletonization(dic,saving=False,savingpath='dict'):
    diclist=list(dic.keys())
    for j in tqdm.trange(len(diclist)):
        fichier=diclist[j]
        
        masks=dic[fichier]['masks']
        mask_number=np.max(masks)
        skeletons=[]
        for i in range(mask_number):
            mask = transfo_bin(masks,i+1) #isolating the i+1-th mask
            mask = cv2.copyMakeBorder(mask,20,20,20,20,cv2.BORDER_CONSTANT,None,value=0)#add a border to the skel
            skel = skeletonize(mask)   #, method='lee'
            # skel = thin(mask, max_num_iter=15)         
            skel = skel[20:np.shape(skel)[0]-20,20:np.shape(skel)[1]-20]
            skeletons.append(skel)
        dic[fichier]['skeleton']=skeletons
    if saving:
        np.savez_compressed(savingpath,dic,allow_pickle=True)


def run_one_dataset_logs_watershed(dic):
    
    my_data='../data/'
    #path of dir
    #directory with the usefull information of the logs, None if there is no.

    #Temporary directories
    #directory for output data from cellpose 
    segments_path = os.path.join(dic,"cellpose_output/")


    #Outputs
    #directory of the processed images (every image has the same pixel size and same zoom)
    #Saving path for the dictionnary
    if os.path.exists(dic):
        for file in os.listdir(dic):
            rmtree(os.path.join(dic, file))
    else:
        os.makedirs(dic)
    saving_path=os.path.join('results',dic,'Main_dictionnary')
    #dimension and scale of all the final data
    #dimension and scale of all the final data
    list_savingpath=os.path.join('results',dic,'masks_list')
    


    
    ''' Parameters'''

    
    #cellpose parameters
    
    cel_model_type='cyto2'# 'cyto''cyto2''nuclei'
    cel_channels=[0,0]  # define CHANNELS to run segementation on grayscale=0, R=1, G=2, B=3; channels = [cytoplasm, nucleus]; nucleus=0 if no nucleus
    cel_diameter_param = 85 #120 parameter to adjust the expected size (in pixels) of bacteria. Incease if cellpose detects too small masks, decrease if it don't detects small mask/ fusion them. Should be around 1 
    cel_flow_threshold = 0.15  #oldparam :0.15   [0.8,0.2]
    cel_cellprob_threshold=0.95 #oldparam :0.95
    cel_flow_threshold_small = 0.95
    cel_cellprob_threshold_small=2
    cel_flow_threshold_big = 0.3
    cel_cellprob_threshold_big=-0.5#oldparam : 0.0 0.4
    cell_gpu=True
    denoise_mod = "denoise_cyto3"
    
    #erasing small masks that have a smaller relative size :
    ratio_erasing_masks=0.2
    
    surf_com_thres=0.5
    boundary_thres=0.3 #threshold to define the boundary of a cell (bigger one if =1, smaller if =0)
    
    
    
    step=0
    ''''''
    
    print("run_cel",step)
    step+=1
    run_cell_simple(os.path.join(my_data, dic),cel_model_type,cel_channels,cel_diameter_param,cel_flow_threshold,cel_cellprob_threshold,segments_path,denoise_mod, gpuval=cell_gpu)
    # run_cell_boundary(my_data+dic,cel_model_type,cel_channels,cel_diameter_param,cel_flow_threshold_small,cel_flow_threshold_big,cel_cellprob_threshold_small,cel_cellprob_threshold_big,segments_path,surf_com_thres,boundary_thres,denoise_mod, gpuval=cell_gpu)

    print("download_dict",step)
    step+=1
    main_dict=download_dict_logs_only(my_data,dic,segments_path,saving=True,savingpath=saving_path)
    
    # main_dict=np.load(saving_path+'.npz', allow_pickle=True)['arr_0'].item()
    
    print("main_parenting",step)
    step+=1
    main_parenting(main_dict)
    
    print("watershed",step)
    step+=1
    mask_watershed(main_dict)
    
    print("centroid_area",step)
    step+=1
    centroid_area(main_dict)
    
    print("clean_masks",step)
    step+=1
    clean_masks(ratio_erasing_masks, main_dict)
    
    print("main_parenting",step)
    step+=1
    main_parenting(main_dict) #re-run in case all masks in a frame are erased
    
    print("skeletonization",step)
    step+=1
    skeletonization(main_dict)
    
    print("construction_mask_list",step)
    step+=1
    construction_mask_list(main_dict,list_savingpath,saving=True,savingpath=saving_path)
    


def run_one_dataset_logs_only(dic):
    
    my_data='../data/'
    #path of dir
    #directory with the usefull information of the logs, None if there is no.

    #Temporary directories
    #directory for output data from cellpose 
    segments_path = os.path.join(dic,"cellpose_output/")


    #Outputs
    #directory of the processed images (every image has the same pixel size and same zoom)
    #Saving path for the dictionnary
    if os.path.exists(dic):
        for file in os.listdir(dic):
            rmtree(os.path.join(dic, file))
    else:
        os.makedirs(dic)
    saving_path=os.path.join('results',dic,'Main_dictionnary')
    #dimension and scale of all the final data
    #dimension and scale of all the final data
    list_savingpath=os.path.join('results',dic,'masks_list')
    


    
    ''' Parameters'''

    
    #cellpose parameters
    
    cel_model_type='cyto2'# 'cyto''cyto2''nuclei'
    cel_channels=[0,0]  # define CHANNELS to run segementation on grayscale=0, R=1, G=2, B=3; channels = [cytoplasm, nucleus]; nucleus=0 if no nucleus
    cel_diameter_param = 85 #120 parameter to adjust the expected size (in pixels) of bacteria. Incease if cellpose detects too small masks, decrease if it don't detects small mask/ fusion them. Should be around 1 
    cel_flow_threshold = 0.15  #oldparam :0.15   [0.8,0.2]
    cel_cellprob_threshold=0.95 #oldparam :0.95
    cel_flow_threshold_small = 0.95
    cel_cellprob_threshold_small=2
    cel_flow_threshold_big = 0.3
    cel_cellprob_threshold_big=-0.5#oldparam : 0.0 0.4
    cell_gpu=True
    denoise_mod = "denoise_cyto3"
    
    #erasing small masks that have a smaller relative size :
    ratio_erasing_masks=0.2
    
    surf_com_thres=0.5
    boundary_thres=0.3 #threshold to define the boundary of a cell (bigger one if =1, smaller if =0)
    
    
    
    step=0
    ''''''
    
    print("run_cel",step)
    step+=1
    run_cell_simple(os.path.join(my_data, dic),cel_model_type,cel_channels,cel_diameter_param,cel_flow_threshold,cel_cellprob_threshold,segments_path,denoise_mod, gpuval=cell_gpu)
    # run_cell_boundary(my_data+dic,cel_model_type,cel_channels,cel_diameter_param,cel_flow_threshold_small,cel_flow_threshold_big,cel_cellprob_threshold_small,cel_cellprob_threshold_big,segments_path,surf_com_thres,boundary_thres,denoise_mod, gpuval=cell_gpu)

    print("download_dict",step)
    step+=1
    main_dict=download_dict_logs_only(my_data,dic,segments_path,saving=True,savingpath=saving_path)
    
    # main_dict=np.load(saving_path+'.npz', allow_pickle=True)['arr_0'].item()
    
    print("main_parenting",step)
    step+=1
    main_parenting(main_dict)
    
    print("centroid_area",step)
    step+=1
    centroid_area(main_dict)

    print("clean_masks",step)
    step+=1
    clean_masks(ratio_erasing_masks, main_dict)
    
    print("main_parenting",step)
    step+=1
    main_parenting(main_dict) #re-run in case all masks in a frame are erased
    
    print("skeletonization",step)
    step+=1
    skeletonization(main_dict)
    
    print("construction_mask_list",step)
    step+=1
    construction_mask_list(main_dict,list_savingpath,saving=True,savingpath=saving_path)
    
    
    
    
def run_cellpose(dic):
    
    my_data='../data/'
    #path of dir
    #directory with the usefull information of the logs, None if there is no.

    #Temporary directories
    #directory for output data from cellpose 
    segments_path = os.path.join(dic,"cellpose_output/")


    #Outputs
    #directory of the processed images (every image has the same pixel size and same zoom)
    #Saving path for the dictionnary
    if os.path.exists(dic):
        for file in os.listdir(dic):
            rmtree(os.path.join(dic, file))
    else:
        os.makedirs(dic)
    


    
    ''' Parameters'''

    
    #cellpose parameters
    
    cel_model_type='cyto2'# 'cyto''cyto2''nuclei'
    cel_channels=[0,0]  # define CHANNELS to run segementation on grayscale=0, R=1, G=2, B=3; channels = [cytoplasm, nucleus]; nucleus=0 if no nucleus
    cel_diameter_param = 85 #120 parameter to adjust the expected size (in pixels) of bacteria. Incease if cellpose detects too small masks, decrease if it don't detects small mask/ fusion them. Should be around 1 
    cel_flow_threshold = 0.15  #oldparam :0.15   [0.8,0.2]
    cel_cellprob_threshold=0.95 #oldparam :0.95
    cel_flow_threshold_small = 0.95
    cel_cellprob_threshold_small=2
    cel_flow_threshold_big = 0.3
    cel_cellprob_threshold_big=-0.5#oldparam : 0.0 0.4
    cell_gpu=True
    denoise_mod = "denoise_cyto3"
    
    
    
    
    step=0
    ''''''
    
    print("run_cel",step)
    step+=1
    run_cell_simple(os.path.join(my_data, dic),cel_model_type,cel_channels,cel_diameter_param,cel_flow_threshold,cel_cellprob_threshold,segments_path,denoise_mod, gpuval=cell_gpu)
    # run_cell_boundary(my_data+dic,cel_model_type,cel_channels,cel_diameter_param,cel_flow_threshold_small,cel_flow_threshold_big,cel_cellprob_threshold_small,cel_cellprob_threshold_big,segments_path,surf_com_thres,boundary_thres,denoise_mod, gpuval=cell_gpu)

    
    
def run_end_preprocess(dic):
    
    my_data='../data/'
    #path of dir
    #directory with the usefull information of the logs, None if there is no.

    #Temporary directories
    #directory for output data from cellpose 
    segments_path = os.path.join(dic,"cellpose_output/")

    saving_path=os.path.join('results',dic,'Main_dictionnary')
    #dimension and scale of all the final data
    #dimension and scale of all the final data
    list_savingpath=os.path.join('results',dic,'masks_list')
    


    
    ''' Parameters'''


    
    #erasing small masks that have a smaller relative size :
    ratio_erasing_masks=0.2
    
    
    step=1
    print("download_dict",step)
    step+=1
    main_dict=download_dict_logs_only(my_data,dic,segments_path,saving=True,savingpath=saving_path)
    
    # main_dict=np.load(saving_path+'.npz', allow_pickle=True)['arr_0'].item()
    
    print("main_parenting",step)
    step+=1
    main_parenting(main_dict)
    
    print("centroid_area",step)
    step+=1
    centroid_area(main_dict)

    print("clean_masks",step)
    step+=1
    clean_masks(ratio_erasing_masks, main_dict)
    
    print("main_parenting",step)
    step+=1
    main_parenting(main_dict) #re-run in case all masks in a frame are erased
    
    print("skeletonization",step)
    step+=1
    skeletonization(main_dict)
    
    print("construction_mask_list",step)
    step+=1
    construction_mask_list(main_dict,list_savingpath,saving=True,savingpath=saving_path)
    


def run_end_preprocess_watershed(dic):
    
    my_data='../data/'
    #path of dir
    #directory with the usefull information of the logs, None if there is no.

    #Temporary directories
    #directory for output data from cellpose 
    segments_path = os.path.join(dic,"cellpose_output/")

    saving_path=os.path.join('results',dic,'Main_dictionnary')
    #dimension and scale of all the final data
    #dimension and scale of all the final data
    list_savingpath=os.path.join('results',dic,'masks_list')
    


    
    ''' Parameters'''


    
    #erasing small masks that have a smaller relative size :
    ratio_erasing_masks=0.2
    
    
    step=1
    print("download_dict",step)
    step+=1
    main_dict=download_dict_logs_only(my_data,dic,segments_path,saving=True,savingpath=saving_path)
    
    # main_dict=np.load(saving_path+'.npz', allow_pickle=True)['arr_0'].item()
    
    print("main_parenting",step)
    step+=1
    main_parenting(main_dict)
    
    print("watershed",step)
    step+=1
    mask_watershed(main_dict)
    
    
    print("centroid_area",step)
    step+=1
    centroid_area(main_dict)

    print("clean_masks",step)
    step+=1
    clean_masks(ratio_erasing_masks, main_dict)
    
    print("main_parenting",step)
    step+=1
    main_parenting(main_dict) #re-run in case all masks in a frame are erased
    
    print("skeletonization",step)
    step+=1
    skeletonization(main_dict)
    
    print("construction_mask_list",step)
    step+=1
    construction_mask_list(main_dict,list_savingpath,saving=True,savingpath=saving_path)

if __name__ == "__main__":
    
    '''Running the different functions'''
    

    Directory= "July6_plate1_xy02"  #the directory you chose to work on    
    run_one_dataset_logs_watershed(Directory)
 
        

