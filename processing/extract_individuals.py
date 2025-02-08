# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:07:51 2023

@author: shawn
"""

import numpy as np

#%% Load data for testing

# # Directory paths
# dictionary_dir = "C:/Users/shawn/OneDrive/Desktop/temp_scripts/Processed_data/dataset/Height/Dic_dir/"
# img_dir = "C:/Users/shawn/OneDrive/Desktop/temp_scripts/Processed_data/dataset/Height/final_data"

# # Load image dictionary, list of all masks, and adjacency matrix for binary trees
# main_dict = np.load(dictionary_dir + "Main_dictionnary.npy",allow_pickle=True).item()
# masks_list = np.load(dictionary_dir + "masks_list.npy",allow_pickle=True)
# link_mat = np.load(dictionary_dir + "Link_matrix.npy")
# adj_mat = np.load(dictionary_dir + "Bool_matrix.npy") #asymmetric adjacency matrix to represent binary tree

#%% Adjacency matrix to dictionary functions

def extract_individuals(adj_mat,save_dir,filename='ROI_dict'):
    """
    Wrapper function. Accepts an adjacency matrix as its input, extracts all 
    individual cell information, and returns and saves a dictionary of ROIs to 
    the specified save directory.
    """
    ROI_dict = {} # initialize dictionary
    roots = get_roots(adj_mat) # get all roots and start new tree traversal for each one 
    starting_num = 1 # initial ROI ID number is 1
    
    for root in roots:
        individuals = get_ROIs(adj_mat,root,ROI_num=starting_num,root=root)[0]
        ROI_dict.update(list2dict(individuals))
        starting_num = int(list(ROI_dict.keys())[-1].split()[-1])+1 # make sure next tree's first ROI is latest ROI ID# + 1
    
    create_children(ROI_dict) # add Children key to each dictionary item
    # np.savez_compressed(save_dir+filename,ROI_dict)
    # print(ROI_dict)
    return ROI_dict
    

def get_roots(adj_mat):
    """ 
    Accepts an adjacency matrix as its input and returns a list of roots of 
    all distinct binary lineage trees.
    """
    roots = ~np.any(adj_mat,axis=0)
    roots = list(np.where(roots)[0])
    return roots


def get_ROIs(adj_mat,current_node,ROI_num=1,root=0):
    """
    Performs a depth first traversal of the binary tree represented in the 
    adjacency matrix input argument and returns 1) a flat list of strings and 
    integers corresponding to each ROI ID, the parent ROI ID, and the mask 
    indices (nodes) for each ROI; 2) the number ID of the last individual ROI 
    encountered.
    """
    ROI = "ROI "+str(ROI_num)
    individuals = [current_node] # initialize list of ROIs
    if current_node==root:
        individuals = [ROI] + individuals # assign the ROI ID for root node
    node_list = np.where(adj_mat[current_node,:])[0] # list the immediate child nodes of the current node
    
    # if there is only one child node, then this node represents a mask belonging to the same individual
    if node_list.size==1: 
        next_node = node_list[0]
        # recursively call the function
        # get a list of all successor nodes for the current node
        # keep track of the ID number of the newest ROI 
        successors, ROI_num = get_ROIs(adj_mat,next_node,ROI_num=ROI_num)
        individuals.extend(successors)
        
    # if there are multiple child nodes, then a division event has occurred    
    elif node_list.size>1: 
        parent_num = ROI_num
        for next_node in node_list: # create a new ROI string and parent string for each child 
            ROI_num+=1
            ROI = "ROI "+str(ROI_num)
            individuals.append(ROI)
            individuals.append('Parent: ROI '+str(parent_num))
            # recursively get all successor nodes for this new individual
            # keep track of newest ROI ID number
            successors, ROI_num = get_ROIs(adj_mat,next_node,ROI_num=ROI_num) 
            individuals.extend(successors)       
    return individuals, ROI_num


def list2dict(individuals):
    """
    Parses the list of ROI strings and mask index integers into an ROI 
    dictionary by creating ROI ID keys and appending the parent IDs and mask 
    IDs to the preceding ROI ID key.
    """
    ROI_dict = {}
    for element in individuals:
        
        # if element is an ROI ID string 
        if isinstance(element, str) and 'Parent' not in element:
            ROI = element
            new_ROI = {ROI: {'Mask IDs': [], 'Parent': ''}}
            ROI_dict.update(new_ROI)
        
        # if element is an parent ROI ID string 
        elif isinstance(element, str) and 'Parent' in element:
            ROI_dict[ROI]['Parent'] = element[element.index(' ')+1:]
            
        # if element is an integer referring to an index in the masks list
        else:
            ROI_dict[ROI]['Mask IDs'].append(element)
    return ROI_dict


def create_children(ROI_dic):
    """
    Add "Children" subkey to complement "Parent" subkey for each ROI. Append empty
    string if none.
    """
    for ROI in ROI_dic.keys():
        ROI_dic[ROI]['Children']=[]
    for ROI in ROI_dic.keys():
        if ROI_dic[ROI]['Parent']!='':
          ROI_dic[ROI_dic[ROI]['Parent']]['Children'].append(ROI)

#%% Obtain full ROI dictionary from the adjacency matrix and save to specified directory

# extract_individuals(adj_mat,dictionary_dir,filename='ROI_dict')


#%% Load and check ROI dictionary
# ROI_dict = np.load(dictionary_dir + "ROI_dict.npy",allow_pickle=True).item()
# print(ROI_dict)
