# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 19:58:44 2023

@author: DELL
"""
import cv2
import tqdm
import numpy as np

from skimage import measure
from skimage.future import graph
from networkx.linalg import adj_matrix
from skimage.segmentation import find_boundaries

from image_io.image_io import imread, imwrite  


    
def shadow_removal(image_path, shadow_mask, segment_lables, region_is_shadows, corrected_image_path):
    """
    This function is used to adjust brightness for shadow - covered areas in an image, as proposed in the paper 
    
    Inputs:
    - image_path: Path of image to be processed for shadow removal. It is assumed that the first 3 channels are ordered as
                  Blue, Green and Red respectively
    - shadow_mask: Shadow mask binary image
    - segment_lables: Lable of superpixel segmentation
    - region_is_shadows: Bool array of each superpixel region is shadows or not
    - corrected_image_path: Path of corrected image to be saved
    
    Outputs:
    - corrected_img: Corrected input image
    
    """
    
    image, geotrans, proj = imread(image_path)
    
    # 计算全局比值
    global_ratios = []
    un_shaded_mask = np.uint8(shadow_mask == 0)
    for i in range(image.shape[0]):
        shaded_light = shadow_mask * image[i, :, :]
        un_shaded_light = un_shaded_mask * image[i, :, :]
        shaded_illumination = np.sum(shaded_light) / np.sum(shadow_mask)
        un_shaded_illumination = np.sum(un_shaded_light) / np.sum(un_shaded_mask)
        global_ratio = ((un_shaded_illumination - shaded_illumination) / shaded_illumination) + 1
        global_ratios.append(global_ratio)
        
    # 获得所有region, region_1.intensity_image代表区域中的像素值; region_1.area代表像素数
    regions = measure.regionprops(segment_lables, intensity_image=image.transpose((1,2,0)))
    region_intensity_images = [r.intensity_image for r in regions]
    region_areas = [r.area for r in regions]
    region_max_intensitys =  [r.max_intensity for r in regions]
    region_min_intensitys =  [r.min_intensity for r in regions]
    
    # lable 转 graph
    graphs = graph.RAG(segment_lables)
    # 计算邻接矩阵，adj[i,j]代表第i和第j个超像素是否相邻
    adjacency_matrix = adj_matrix(graphs).todense()
    adjacency_matrix_np = np.array(adjacency_matrix)
    
    # 初始阴影重建图像，即阴影区域像素值为0
    corrected_img = np.multiply(image, np.array([(shadow_mask==False)]*image.shape[0]))
    
    # 阴影region的索引
    region_shadow_indexs = np.where(region_is_shadows)
    for region_shadow_index in tqdm.tqdm(region_shadow_indexs[0]):
        
        # 与image相同尺寸的region，即region之外的像素为0
        image_region = np.multiply(image, np.array([(segment_lables==(region_shadow_index+1))]*image.shape[0]))
        
        # 该阴影region的像素、面积、照度
        region_shadow_intensity_image = region_intensity_images[region_shadow_index]
        
        region_shadow_area = region_areas[region_shadow_index]
        region_shadow_max_intensity = region_max_intensitys[region_shadow_index]
        region_shadow_min_intensity = region_min_intensitys[region_shadow_index]
        shaded_illumination_bands = []
        for i in range(region_shadow_intensity_image.shape[2]):
            shaded_illumination_bands.append(np.sum(region_shadow_intensity_image[:,:,i])/region_shadow_area)
        # 该阴影region的直方图 -> 为计算巴氏系数服务
        region_shadow_intensity_image_hist = []
        for i in range(image.shape[0]):
            region_shadow_intensity_image_hist.append(cv2.calcHist([region_shadow_intensity_image], 
                                                                   [i], None, [256], [0, 256]))
        region_shadow_intensity_image_hist = np.sum(region_shadow_intensity_image_hist,0)
        
        # 该region的相邻region的index
        region_shadow_adj_indexs = np.where(adjacency_matrix_np[region_shadow_index])
        un_shaded_illuminations = []
        un_shaded_areas = []
        bhattacharyya_coefficients = []
        
        unshaped_region_intensity_images = np.full((1,4), np.nan).astype(np.float32)
        
        for region_shadow_adj_index in region_shadow_adj_indexs[0]:
            # 若该相邻region为非阴影的话，计算照度
            if not region_is_shadows[region_shadow_adj_index]:
                region_intensity_image = region_intensity_images[region_shadow_adj_index]
                region_area = region_areas[region_shadow_adj_index]
                region_max_intensity = region_max_intensitys[region_shadow_adj_index]
                region_min_intensity = region_min_intensitys[region_shadow_adj_index]
                
                region_intensity_image_0_nan = region_intensity_image.copy()
                region_intensity_image_0_nan = region_intensity_image_0_nan.astype(np.float32)
                region_intensity_image_0_nan[region_intensity_image_0_nan==0] = np.nan
                region_intensity_image_0_nan = region_intensity_image_0_nan.reshape(-1,4)
                unshaped_region_intensity_images = np.concatenate((unshaped_region_intensity_images, 
                                                                   region_intensity_image_0_nan),0)
                
                
                un_shaded_illumination_region_bands = []
                for i in range(region_intensity_image.shape[2]):
                    un_shaded_illumination_region_bands.append(np.sum(region_intensity_image[:,:,i]))
                un_shaded_illuminations.append(un_shaded_illumination_region_bands)
                un_shaded_areas.append(region_area)
                
                # 该region的直方图 -> 为计算巴氏系数服务
                region_intensity_image_hist = []
                # 计算直方图之前先将像素值拉伸到阴影region的最值区间
                region_intensity_image = (region_intensity_image - region_min_intensity)/\
                    (region_max_intensity-region_min_intensity)*\
                        (region_shadow_max_intensity-region_shadow_min_intensity)+\
                            region_shadow_min_intensity
                region_intensity_image = np.uint8(region_intensity_image)
                for i in range(image.shape[0]):
                    region_intensity_image_hist.append(cv2.calcHist([region_intensity_image], 
                                                                    [i], None, [256], [0, 256]))
                region_intensity_image_hist = np.sum(region_intensity_image_hist,0)
                bhattacharyya_coefficient = cv2.compareHist(region_shadow_intensity_image_hist, 
                                                            region_intensity_image_hist, 
                                                            method=cv2.HISTCMP_BHATTACHARYYA)
                bhattacharyya_coefficients.append(bhattacharyya_coefficient)
        
        # 每个region的像素权重
        weights = []
        bhattacharyya_coefficients = np.array(bhattacharyya_coefficients)
        bhattacharyya_coefficients = 1 - bhattacharyya_coefficients
        if np.sum(bhattacharyya_coefficients)==0:
            bhattacharyya_coefficients = np.ones(bhattacharyya_coefficients.shape,np.uint8)
        for bhattacharyya_coefficient in bhattacharyya_coefficients:
            weights.append(bhattacharyya_coefficient/np.sum(bhattacharyya_coefficients)*bhattacharyya_coefficients.size)
        weights = np.array(weights)
        
        # 临近region的像素总数
        un_shaded_areas = np.array(un_shaded_areas)
        
        un_shaded_illumination_bands = []
        un_shaded_illuminations = np.array(un_shaded_illuminations)
        # 循环波段
        for i in range(un_shaded_illuminations.shape[1]):
            un_shaded_illumination_i = np.mean(un_shaded_illuminations[:,i] * weights / un_shaded_areas)
            un_shaded_illumination_bands.append(un_shaded_illumination_i)
        
        un_shaded_illumination_bands = np.array(un_shaded_illumination_bands)
        shaded_illumination_bands = np.array(shaded_illumination_bands)
        # 计算阴影重建比
        ratio_bands = ((un_shaded_illumination_bands - shaded_illumination_bands) / shaded_illumination_bands) + 1
        
        unshaped_region_intensity_images_premax = np.nanpercentile(unshaped_region_intensity_images,90,0)
        for i in range(unshaped_region_intensity_images_premax.shape[0]):
            # 如果比值大于全局比值，则将比值赋值为全局比值
            if ratio_bands[i] > global_ratios[i]:
                ratio_bands[i] = global_ratios[i]
        
        # 排除比值小于1的情况
        if np.min(ratio_bands) < 1:
            ratio_bands[ratio_bands<1] = 1
        
        # 对 numpy 矩阵进行复制并使其扩充, 以满足矩阵相乘
        ratio_bands = np.tile(ratio_bands.reshape(-1,1,1),(1,image_region.shape[1],image.shape[2]))
        # 阴影重建
        image_region_correct = np.clip(image_region * ratio_bands, 0, 255).astype(np.uint8)
        corrected_img += image_region_correct
        
        region_intensity_image = region_intensity_images[region_shadow_index]
        ratio_bands = ratio_bands.transpose((1,2,0))[:region_intensity_image.shape[0],
                                                     :region_intensity_image.shape[1],
                                                     :]
        region_intensity_image_correct = np.clip(region_intensity_image * ratio_bands, 0, 255)
        region_intensity_images[region_shadow_index] = region_intensity_image_correct
        # 阴影region重建完毕，修改属性为非阴影，可为相邻的阴影重建服务
        region_is_shadows[region_shadow_index] = False
        
    
    # 对阴影的边界区域进行均值滤波，以缓解半影区域的过补偿和未补偿
    corrected_img = corrected_img.transpose((1,2,0))
    corrected_img_blur = cv2.blur(corrected_img,(5,5))
    shadow_mask_boundaries = find_boundaries(shadow_mask)
    # 阴影边界3像素缓冲区认为是半影区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    shadow_mask_boundaries = cv2.dilate(np.uint8(shadow_mask_boundaries), kernel)
    shadow_mask_boundaries = shadow_mask_boundaries.reshape(shadow_mask_boundaries.shape+(1,))
    shadow_mask_boundaries = np.tile(shadow_mask_boundaries,(1,1,4))
    final_image = shadow_mask_boundaries*corrected_img_blur + (1-shadow_mask_boundaries)*corrected_img
    final_image = final_image.transpose((2,0,1))
    # 保存结果
    imwrite(final_image, geotrans, proj, corrected_image_path)
    return corrected_img, final_image

    