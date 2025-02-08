#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:34:34 2023

@author: c.soubrier
"""
import re
import os

import numpy as np
import matplotlib.pyplot as plt
from cellpose import plot, utils
from PIL import Image
from multiprocessing import Pool

import segmentation as seg
import final_graph as fg
import extract_individuals as exi


Directory= "July6_plate1_xy02/" 


''' Parameters '''

#minimal threshold to consider a daughter-mother relation
final_thresh=0.75  #0.8

#threshold to consider that there is a division and not just a break in the ROI
thres_min_division=0.7 #0.7 

#minimum number of frames in an ROI
min_len_ROI=3 


''' Output'''

#Dictionnary of the ROIs with the ROI name as entry and as argument : Parent ROI (string, if not ''), Child1 ROI (string, if not ''), Child2 ROI (string, if not ''), list of masks (int), ROI index, Roi color_index



''' Functions'''

def filter_good_ROI_dic(ROI_dic,min_number):
    newdic={}
    for ROI in ROI_dic.keys():
        if len(ROI_dic[ROI]['Mask IDs'])>=min_number:
            newdic[ROI]=ROI_dic[ROI]
    termination=False
    while not termination:
        termination=True
        keys=list(newdic.keys())
        for ROI in keys:
            parent=ROI_dic[ROI]['Parent']
            if parent!='' and parent not in keys:
                termination=False
                newdic[parent]=ROI_dic[parent]
    termination=False
    while not termination:
        termination=True
        keys=list(newdic.keys())
        for ROI in keys:
            children=ROI_dic[ROI]['Children']
            if children!=[] and len(children)<=2:
                for i in [0,1]:
                    if children[i] not in keys:
                        termination=False
                        newdic[children[i]]=ROI_dic[children[i]]
            elif len(children)>2:
                print('More then 2 children '+ ROI, ROI_dic[ROI]['Children'])
                for child in ROI_dic[ROI]['Children']:
                    if child in newdic.keys():
                        newdic[child]['Parent']=''
                ROI_dic[ROI]['Children']=[]

    return newdic

def ROI_index(ROI_dic):
    count=1
    colorcount=1
    for ROI in ROI_dic.keys():
        ROI_dic[ROI]['color_index']=colorcount
        colorcount+=1
        if ROI_dic[ROI]['Parent']=='':
          ROI_dic[ROI]['index']=str(count)+'/'
          count+=1
        else:
            ROI_dic[ROI]['index']=''
    for ROI in ROI_dic.keys():
        update_ROI_index(ROI,ROI_dic)
        
def create_children(ROI_dic):
    for ROI in ROI_dic.keys():
        ROI_dic[ROI]['Children']=[]
    for ROI in ROI_dic.keys():
        if ROI_dic[ROI]['Parent']!='':
          ROI_dic[ROI_dic[ROI]['Parent']]['Children'].append(ROI)
              


def update_ROI_index(ROI,ROI_dic,terminal=True):
    if ROI_dic[ROI]['index']=='':
        parent=ROI_dic[ROI]['Parent']
        index_parent=update_ROI_index(parent,ROI_dic,terminal=False)
        if ROI==ROI_dic[parent]['Children'][0]:
            new_index=index_parent+'0'
        else:
            new_index=index_parent+'1'
        ROI_dic[ROI]['index']=new_index
        if not terminal:
            return new_index
    elif not terminal:
        return ROI_dic[ROI]['index']
    
    
    
def intensity_lineage(index):
    res=re.findall(r'\d+\.\d+|\d+',index)
    if len(res)==1:
        return int(res[0])
    else:
        num,suc=res[0],res[1]
        count=int(num)
        for i in range(len(suc)):
            count+= 2**(-i-1)*(1/2-int(suc[i]))
        return count




def plot_lineage_tree(ROI_dic,masks_list,main_dic,maskcol,directory):
    
    for ROI in ROI_dic.keys():
        if ROI_dic[ROI]['Parent']!='':
            parent=ROI_dic[ROI]['Parent']
            value1=intensity_lineage(ROI_dic[parent]['index'])
            value2=intensity_lineage(ROI_dic[ROI]['index'])
            point1=ROI_dic[parent]['Mask IDs'][-1]
            point2=ROI_dic[ROI]['Mask IDs'][0]
            t1=main_dic[masks_list[point1][2]]['time']
            t2=main_dic[masks_list[point2][2]]['time']
            plt.plot([t1,t2],[value1,value2],color='k')
            value1,t1=value2,t2
            color=ROI_dic[ROI]['color_index']
        else:
            value1=intensity_lineage(ROI_dic[ROI]['index'])
            point1=ROI_dic[ROI]['Mask IDs'][0]
            t1=main_dic[masks_list[point1][2]]['time']
            color=ROI_dic[ROI]['color_index']
        
        point2=ROI_dic[ROI]['Mask IDs'][-1]
        t2=main_dic[masks_list[point2][2]]['time']
        len_col=len(maskcol)
        col=maskcol[int(color%len_col)]
        
        plot_col=(col[0]/255,col[1]/255,col[2]/255)
        plt.plot([t1,t2],[value1,value1],color=plot_col)
        plt.title('Lineage tree : '+directory)
    plt.show()
    
def extract_roi_list_from_dic(ROI_dic,masks_list):
    newlist=np.zeros((len(masks_list),3),dtype=object)
    for ROI in ROI_dic.keys():
        color=ROI_dic[ROI]['color_index']
        index=ROI_dic[ROI]['index']
        for i in ROI_dic[ROI]['Mask IDs']:
            newlist[i,0]=color
            newlist[i,1]=index
            newlist[i,2]=ROI
    return newlist




def plot_image_one_ROI(ROI,ROI_dic,masks_list,dic):
    plotlist=ROI_dic[ROI]['Mask IDs']
    Roi_index=ROI_dic[ROI]['index']
    for maskindex in plotlist:
        file=masks_list[maskindex][2]
        index=masks_list[maskindex][3]
        img=np.array(Image.open(dic[file]['adress']))[:,:,1]
        mask=(dic[file]['masks']==index).astype(int)
        mask_RGB =plot.mask_overlay(img,mask)
        plt.title('time : '+str(dic[file]['time']))
        plt.imshow(mask_RGB)
        centroid=dic[file]['centroid'][index-1]
        center=dic[file]['centerlines'][index-1]
        plt.annotate(Roi_index, centroid[::-1], xytext=[10,0], textcoords='offset pixels', color='dimgrey')
        if len(center)>1:
            plt.plot(center[:,1],center[:,0], color='k')
        plt.show()
        
        


def plot_image_lineage_tree(ROI_dic,masks_list,dic,maskcol,indexlist,directory,show=True):
    newdirect='../masks/'+directory
    if os.path.exists(newdirect):
        for file in os.listdir(newdirect):
            os.remove(os.path.join(newdirect, file))
    else:
        os.makedirs(newdirect)
    
    fichier=list(dic.keys())[0]
    while dic[fichier]['child']!='':
        plt.figure()
        # plot image with masks overlaid
        img=np.array(Image.open(dic[fichier]['adress']))[:,:,1]
        # plt.imshow(img,cmap='gray')
        # plt.title(' time : '+str(dic[fichier]['time']))
        # plt.show()
        masks=dic[fichier]['masks']
        masknumber=np.max(masks)
        col_ind_list=np.zeros(masknumber,dtype=np.int32)
        roi_ind_list=[]
        for i in range(masknumber):
            roi_ind_list.append([])
        for i in range(masknumber):
            elem=indexlist[dic[fichier]["mask_list"][i]]
            col_ind_list[i]=elem[0]
            roi_ind_list[i]=str(elem[1])
        
        
        len_col=len(maskcol)
        for i in range(masknumber):
            if roi_ind_list[i]=='0':
                col_ind_list[i]=0
            else:
                col_ind_list[i]=col_ind_list[i]%len_col+1
        masks=seg.update_masks(masks,col_ind_list)
        
        colormask=np.array(maskcol)
        mask_RGB = plot.mask_overlay(seg.renorm_img(img),masks,colors=colormask)#image with masks
        plt.imshow(mask_RGB)
        
        # plot the centroids and the centerlines
        centr=dic[fichier]['centroid']
        outlines=dic[fichier]['outlines']
        for i in range(len(centr)):
            #centroids
            plt.plot(centr[i,1], centr[i,0], color='k',marker='o')
            plt.annotate(roi_ind_list[i], centr[i,::-1], xytext=[10,0], textcoords='offset pixels', color='dimgrey')
            skel = utils.outlines_list(dic[fichier]['skeleton'][i], multiprocessing = False)[0]
            plt.plot(outlines[i][:,1], outlines[i][:,0], color='r',linewidth=0.5)
            plt.plot(skel[:,0], skel[:,1], color='w',linewidth=0.5)
        
        main_centroid=dic[fichier]['main_centroid']
        plt.plot(main_centroid[1], main_centroid[0], color='w',marker='o')
            
        #plot the displacement of the centroid between two images
        plt.title(f'{directory} at time {dic[fichier]["time"]}')
        plt.savefig(os.path.join(newdirect, fichier+'.jpg'), format='jpg', dpi=400)
        if show:
            plt.show()
        else:
            plt.close()
        
        fichier=dic[fichier]['child']
    
    plt.figure()
    # plot image with masks overlaid
    img=np.array(Image.open(dic[fichier]['adress']))[:,:,1]
    # plt.imshow(img,cmap='gray')
    # plt.title(' time : '+str(dic[fichier]['time']))
    # plt.show()
    masks=dic[fichier]['masks']
    masknumber=np.max(masks)
    col_ind_list=np.zeros(masknumber,dtype=np.int32)
    roi_ind_list=[]
    for i in range(masknumber):
        roi_ind_list.append([])
    for i in range(masknumber):
        elem=indexlist[dic[fichier]["mask_list"][i]]
        col_ind_list[i]=elem[0]
        roi_ind_list[i]=str(elem[1])
    
    
    len_col=len(maskcol)
    for i in range(masknumber):
        if roi_ind_list[i]=='0':
            col_ind_list[i]=0
        else:
            col_ind_list[i]=col_ind_list[i]%len_col+1
    masks=seg.update_masks(masks,col_ind_list)
    colormask=np.array(maskcol)
    mask_RGB = plot.mask_overlay(img,masks,colors=colormask)#image with masks
    plt.imshow(mask_RGB)
    
    # plot the centroids and the centerlines
    centr=dic[fichier]['centroid']
    outlines=dic[fichier]['outlines']
    for i in range(len(centr)):
        #centroids
        plt.plot(centr[i,1], centr[i,0], color='k',marker='o')
        plt.annotate(roi_ind_list[i], centr[i,::-1], xytext=[10,0], textcoords='offset pixels', color='dimgrey')
        skel = utils.outlines_list(dic[fichier]['skeleton'][i],multiprocessing=False)[0]
        plt.plot(outlines[i][:,1], outlines[i][:,0], color='r',linewidth=0.5)
        plt.plot(skel[:,0], skel[:,1], color='w',linewidth=0.5)
    
    main_centroid=dic[fichier]['main_centroid']
    plt.plot(main_centroid[1], main_centroid[0], color='w',marker='o')
    plt.title(f'{directory} at time {dic[fichier]["time"]}')
    plt.savefig(os.path.join(newdirect, fichier+'.jpg'), format='jpg', dpi=400)
    #plot the displacement of the centroid between two images
    if show:
        plt.show()
    else:
        plt.close()



    
def rank_subtrees(ROI_dic,ROI_min_number):
    max_root=0
    for ROI in ROI_dic.keys():
        root_index=round(ROI_dic[ROI]['index'])
        if root_index>max_root:
            max_root=root_index
    rank=np.zeros(max_root)
    for ROI in ROI_dic.keys():
        root_index=round(ROI_dic[ROI]['index'])
        rank[root_index-1]+=1
    order=np.argsort(rank)[::-1]
    stop_number=-1
    for i in range(len(order)):
        if root_index[order[i]]>=ROI_min_number:
            order[i]+=1
        else:
            stop_number=i
            break
    return order[:stop_number]
    


def detect_bad_div(ROI_dic,linmatrix,masks_list,thres,thres_min):
    indexlist=extract_roi_list_from_dic(ROI_dic,masks_list)
    for ROI in list(ROI_dic.keys()):
        if ROI_dic[ROI]['Parent']=='':
            first_elem=ROI_dic[ROI]['Mask IDs'][0]
            elem=first_elem
            termination=False
            while not termination and elem>0:
                elem-=1
                if linmatrix[first_elem,elem]>thres and linmatrix[elem,first_elem]<thres_min and indexlist[elem][1]!='0' and indexlist[elem][1]!=0:
                    
                    if ROI_dic[indexlist[elem][2]]['Mask IDs'][-1]<first_elem:
                    
                        print('regluing' +ROI,elem)
                        
                        if ROI_dic[indexlist[elem][2]]['Children']==[]:
                            newindex=indexlist[elem][1]+'0'
                            #print(indexlist[elem][1],newindex)
                            ROI_dic[ROI]['Parent']=indexlist[elem][2]
                            ROI_dic[indexlist[elem][2]]['Children'].append(ROI)
                            
                            change_root_index(ROI,ROI_dic,newindex,indexlist)
                            
                        elif len(ROI_dic[indexlist[elem][2]]['Children'])==1:
                            newindex=indexlist[elem][1]+'1'
                            #print(indexlist[elem][1],newindex)
                            ROI_dic[indexlist[elem][2]]['Children'].append(ROI)
                            ROI_dic[ROI]['Parent']=indexlist[elem][2]
                            
                            change_root_index(ROI,ROI_dic,newindex,indexlist)
                            
                        else:
                            print('Error in re-gluing ROI',ROI,ROI_dic[indexlist[elem][2]])
                        termination=True
    return indexlist






def change_root_index(ROI,ROI_dic,newindex,indexlist):
    var=re.findall(r'\d+\.\d+|\d+',ROI_dic[ROI]['index'])
    if len(var)==1:
        index=newindex
    else:
        index=newindex+var[1]
    ROI_dic[ROI]['index']=index
    for elem in ROI_dic[ROI]['Mask IDs']:
        indexlist[elem][1]=index
    for child in ROI_dic[ROI]['Children']:
        change_root_index(child,ROI_dic,newindex,indexlist)


def manually_regluing(direc,ROIdict,indexlistname,parent,child,division=True):
    ROI_dict=np.load(direc+ROIdict,allow_pickle=True)['arr_0'].item()
    indexlist=np.load(direc+indexlistname,allow_pickle=True)['arr_0']
    parentROI=''
    childROI=''
    for ROI in ROI_dict.keys():
        if ROI_dict[ROI]['index']==parent:
            parentROI=ROI
        elif ROI_dict[ROI]['index']==child:
            childROI=ROI
            
    if parentROI!=''and childROI!='':
        if division:
            if ROI_dict[parentROI]['Children']==[]:
                ROI_dict[parentROI]['Children'].append(childROI)
                ROI_dict[childROI]['Parent']=parentROI
                change_root_index(childROI,ROI_dict,parent+'0',indexlist)
            elif len(ROI_dict[parentROI]['Children'])==1:
                ROI_dict[parentROI]['Children'].append(childROI)
                ROI_dict[childROI]['Parent']=parentROI
                change_root_index(childROI,ROI_dict,parent+'1',indexlist)
            else:
                raise ValueError('already 2 children for '+parent)
        else:
            if not ROI_dict[parentROI]['Children']==[]:
                raise ValueError('already children for '+parent)
            else :
                change_root_index(childROI,ROI_dict,parent,indexlist)
                color=ROI_dict[parentROI]['color_index']
                index=ROI_dict[parentROI]['index']
                for grand_children in ROI_dict[childROI]['Children']:
                    ROI_dict[grand_children]['Parent']=parentROI
                    ROI_dict[parentROI]['Children'].append(grand_children)
                for mask in ROI_dict[childROI]['Mask IDs']:
                    ROI_dict[parentROI]['Mask IDs'].append(mask)
                    indexlist[mask,0]=color
                    indexlist[mask,1]=index
                    indexlist[mask,2]=parentROI
    else:
        raise NameError('No ROI with following index :'+(parentROI=='')*parent+' '+(childROI=='')*child)
    np.savez_compressed(direc+ROIdict,ROI_dict,allow_pickle=True)
    np.savez_compressed(direc+indexlistname,indexlist,allow_pickle=True)
            
                
    
    
    
    
    
def run_whole_lineage_tree(direc,thres=final_thresh,min_number=min_len_ROI,thresmin=thres_min_division,show=True):
    
    
    dicname='Main_dictionnary.npz'

    listname='masks_list.npz'

    ROIdict='ROI_dict.npz'

    linmatname='non_trig_Link_matrix.npy'

    boolmatname="Bool_matrix.npy"

    linkmatname='Link_matrix.npy'

    indexlistname='masks_ROI_list.npz'
    
    path = os.path.join('results', direc)
    
    colormask=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,204,130],[130,255,204],[130,0,255],[130,204,255]]

    
    masks_list=np.load(os.path.join(path, listname), allow_pickle=True)['arr_0']
    main_dict=np.load(os.path.join(path, dicname), allow_pickle=True)['arr_0'].item()
    Bool_matrix=np.load(os.path.join(path, boolmatname))

    
    ROI_dict=exi.extract_individuals(Bool_matrix, direc)
    
    linmatrix=np.load(os.path.join(path, linmatname))
    
    newdic=filter_good_ROI_dic(ROI_dict,min_number)
    
    ROI_index(newdic)
    
    indexlist=detect_bad_div(newdic,linmatrix,masks_list,thres,thresmin)
    if show:
        plot_lineage_tree(newdic,masks_list,main_dict,colormask,direc)
    plot_image_lineage_tree(newdic,masks_list,main_dict,colormask,indexlist,direc,show=show)
    np.savez_compressed(os.path.join(path, ROIdict),newdic,allow_pickle=True)
    np.savez_compressed(os.path.join(path, indexlistname),indexlist,allow_pickle=True)
    
    
    os.remove(os.path.join(path, linmatname))
    os.remove(os.path.join(path, boolmatname))
    os.remove(os.path.join(path, linkmatname))
    
 
def main(direc):
    print(direc)
    seg.run_one_dataset_logs_watershed(direc)
    fg.Final_lineage_tree(direc)
    run_whole_lineage_tree(direc,show=False)# 



    
if __name__ == "__main__":
    os.chdir('..')
    Directory = "July6_plate1_xy02" 
    main(Directory)
   

    