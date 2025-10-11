#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:08:29 2023

@author: c.soubrier
"""
import sys
import cv2
import os
import numpy as np


def create_video(direc):

    image_folder = os.path.join('..', 'masks', direc)
    images = [img for img in os.listdir(image_folder)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    print (height, width)

    video = cv2.VideoWriter(os.path.join('..', 'masks', direc+'_video.avi'), 0,1, (width,height))#cv2.VideoWriter.fourcc(*'XVID')

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()



if __name__ == "__main__":
    Directory = "July6_plate1_xy02" 
    create_video(Directory)