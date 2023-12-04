# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 19:58:44 2023

@author: DELL
"""

import numpy as np

from skimage import measure
from skimage.future import graph
from skimage.segmentation import slic
from skimage.exposure import rescale_intensity
from skimage.color import lab2lch, rgb2lab
from skimage.filters import threshold_multiotsu

from image_io.image_io import imread, imwrite  
from utils.graph_merge_func import weight_mean_color, merge_mean_color
    

    
def shadow_detection(image_path, shadow_path):
    """
    This function is used to detect shadow - covered areas in an image, as proposed in the paper 
    
    Inputs:
    - image_path: Path of image to be processed for shadow removal. It is assumed that the first 3 channels are ordered as
                  Blue, Green and Red respectively
    - shadow_path: Path of shadow mask to be saved
    
    Outputs:
    - shadow_mask: Shadow mask for input image
    
    """
    
    image, geotrans, proj = imread(image_path)
    print(image_path)
    # (c,w,h) -> (w,h,c)
    image = image.transpose((1,2,0))
    # nir = image[:,:,-1]
    # bgrnir -> rgb
    image_rgb = image[:,:,:3][...,::-1]
    
    # 从 RGB 模型到 CIELAB 模型的图像转换，以便从强度信息中分离颜色
    image_lab = rgb2lab(image_rgb)
    
    # 从 CIELAB 颜色空间转换为其极坐标表示 CIELCh，因此我们可以使用色调通道来利用阴影具有较大色调值的事实
    image_lch = np.float32(lab2lch(image_lab))
    
    # 改进的光谱比SR计算，使用 CIELCh 而不是 HSI 色彩空间
    l_norm = rescale_intensity(image_lch[:, :, 0], out_range = (0, 1))
    h_norm = rescale_intensity(image_lch[:, :, 2], out_range = (0, 1))
    # nir_norm = rescale_intensity(nir, out_range = (0, 1))
    specthem_ratio = (h_norm + 1) / (l_norm + 1)
    # # 计算了Specthem Ratio 图像每个像素的自然对数，以将原始值压缩到更窄的范围内
    # specthem_ratio = np.log(specthem_ratio)
    
    print("Spectral ratio calculation completed.")
    
    # 根据每200个像素一个超像素来确定初步超像素数目
    n_segments = int(image_rgb.shape[0] * image_rgb.shape[1] / 200)
    segment_lables = slic(image_rgb, n_segments=n_segments, compactness=10, sigma=1,
                         start_label=1)
    
    print("Superpixel segmentation completed.")
    
    # 对相邻均值相似的超像素合并
    rag = graph.rag_mean_color(image_rgb, segment_lables)
    segment_lables_merge = graph.merge_hierarchical(
        labels = segment_lables, 
        rag = rag,
        thresh = 40, # 合并由权重小于thresh的边连接的区域
        rag_copy = False, # bool如果设置，RAG在修改之前被复制
        in_place_merge = True, # 如果设置，则节点合并原地操作，节省内存
        merge_func = merge_mean_color, # 在合并两个节点之前调用此函数
        weight_func = weight_mean_color # 计算与合并节点相邻的节点的新权重的函数
        )
    
    print("Superpixel merge completed.")
    
    # segment_lables在merge后由从1标记改为了从0标记
    segment_lables_merge = segment_lables_merge + 1
    # 来存储每个超像素的MBI均值
    specthem_ratio_means = []
    # 获得所有region, region_1.intensity_image代表区域中的像素值; region_1.area代表像素数
    regions = measure.regionprops(segment_lables_merge, intensity_image=specthem_ratio)
    specthem_ratio_means = np.array([r.mean_intensity for r in regions])
    
    # 将均值投射到对应像素位置/之所以-1是因为segments_slic是从1开始标记的
    specthem_ratio_mean = specthem_ratio_means[segment_lables_merge-1]
    
    print("Spectral ratio reconstruction completed.")
    
    # # OUST计算阈值,不如多级OUST
    # threshold_value = threshold_otsu(ISI_mean)
    # 自动多级全局阈值确定
    threshold_value = threshold_multiotsu(specthem_ratio_means, classes=5)[-1]
    
    print("Automatic multi-level global threshold calculation completed.")
    
    # 每个region是否为阴影
    region_is_shadows = specthem_ratio_means >= threshold_value
    
    # 每个像素是否为阴影
    shadow_mask = np.uint8(specthem_ratio_mean >= threshold_value)
    
    imwrite(shadow_mask, geotrans, proj, shadow_path)
    imwrite(shadow_mask*255, geotrans, proj, shadow_path.replace(".tif","_255.tif"))
    return shadow_mask, segment_lables_merge, region_is_shadows