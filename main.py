# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:31:52 2023

@author: DELL
"""
import os
from shadow_detection.shadow_detection import shadow_detection
from shadow_removal.shadow_removal import shadow_removal

def shadow_detection_removal(image_path):
    shadow_path = image_path.replace(".tif","_shadow_detection.tif")
    shadow_dir = os.path.dirname(shadow_path)
    if not os.path.exists(shadow_dir): os.makedirs(shadow_dir)
    
    reconstruction_path = image_path.replace(".tif","_shadow_reconstruction.tif")
    reconstruction_dir = os.path.dirname(reconstruction_path)
    if not os.path.exists(reconstruction_dir): os.makedirs(reconstruction_dir)
    
    
    shadow_mask, segment_lables, region_is_shadows = shadow_detection(
        image_path, 
        shadow_path
        )
    
    corrected_img, final_image = shadow_removal(
        image_path, 
        shadow_mask, 
        segment_lables, 
        region_is_shadows, 
        reconstruction_path
        )
    
if __name__ == '__main__':
    
    
    image_path = "test_data/demo.tif"
    shadow_detection_removal(image_path)