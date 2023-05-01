#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:57:38 2021

@author: Mathew
"""

from skimage.io import imread
import os
import pandas as pd
from picasso import render
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from skimage import filters,measure
from skimage.filters import threshold_local



# Set to 1 for which process is required:


# For autofluorescence, an unlabelled sample is rweqquired:
calculate_autof=0

# For bleedthrough a donor only labelled image is required:
calculate_bleedthrough=0           

# Calculare direct-excitation of acceptor. For this an acceptor only labelled image is required. 
calculate_direct_excitation=0


# FRET analysis:

FRET_analysis=1

# Which files to analyse:

# Where to save the results
root_path="/Users/Mathew/Documents/Current analysis/FRET_Takeshi_221005/PSD95-eGFP_SAP102-mKO2_211103/"

# Donor

donor_path="/Users/Mathew/Documents/Current analysis/FRET_Takeshi_221005/PSD95-eGFP_SAP102-mKO2_211103/GFP channel/GFP_PSD95-eGFP_SAP102-mKO2_DG_1.tif"
acceptor_path="/Users/Mathew/Documents/Current analysis/FRET_Takeshi_221005/PSD95-eGFP_SAP102-mKO2_211103/mKO2 channel/mKO2_PSD95-eGFP_SAP102-mKO2_DG_1.tif"
FRET_path="/Users/Mathew/Documents/Current analysis/FRET_Takeshi_221005/PSD95-eGFP_SAP102-mKO2_211103/FRET channel/FRET_PSD95-eGFP_SAP102-mKO2_DG_1.tif"


# Values for autofluorescence etc. 

autof_donor=113
autof_acceptor=107
autof_FRET=110
crosstalk=0.0677
direct_excitation=0.03

# autof_donor=10
# autof_acceptor=10
# autof_FRET=10
# crosstalk=0.0677
# direct_excitation=0.03

def load_image(toload):
    
    image=imread(toload)
    
    return image

def z_project(image):
    
    mean_int=np.mean(image,axis=0)
    
    return mean_int

# Subtract background:
def subtract_bg(image):
    background = threshold_local(image, 11, offset=np.percentile(image, 1), method='median')
    bg_corrected =image - background
    return bg_corrected

def threshold_image_otsu(input_image):
    threshold_value=filters.threshold_otsu(input_image)  
    
    # threshold_value=input_image.mean()+3*input_image.std()
    print(threshold_value)
    binary_image=input_image>threshold_value
    
    return threshold_value,binary_image

def threshold_image_standard(input_image,thresh):
    
    binary_image=input_image>thresh
    
    return binary_image

# Threshold image using otsu method and output the filtered image along with the threshold value applied:

def threshold_image_fixed(input_image,threshold_number):
    threshold_value=threshold_number   
    binary_image=input_image>threshold_value
    
    return threshold_value,binary_image

# Label and count the features in the thresholded image:
def label_image(input_image):
    labelled_image=measure.label(input_image)
    number_of_features=labelled_image.max()
    
    return number_of_features,labelled_image

# Function to show the particular image:
def show(input_image,color=''):
    if(color=='Red'):
        plt.imshow(input_image,cmap="Reds")
        plt.show()
    elif(color=='Blue'):
        plt.imshow(input_image,cmap="Blues")
        plt.show()
    elif(color=='Green'):
        plt.imshow(input_image,cmap="Greens")
        plt.show()
    else:
        plt.imshow(input_image)
        plt.show() 


# Take a labelled image and the original image and measure intensities, sizes etc.
def analyse_labelled_image(labelled_image,original_image):
    measure_image=measure.regionprops_table(labelled_image,intensity_image=original_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length','mean_intensity','max_intensity'))
    measure_dataframe=pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

# This is to look at coincidence purely in terms of pixels

def coincidence_analysis_pixels(binary_image1,binary_image2):
    pixel_overlap_image=binary_image1&binary_image2         
    pixel_overlap_count=pixel_overlap_image.sum()
    pixel_fraction=pixel_overlap_image.sum()/binary_image1.sum()
    
    return pixel_overlap_image,pixel_overlap_count,pixel_fraction

# Look at coincidence in terms of features. Needs binary image input 

def feature_coincidence(binary_image1,binary_image2):
    number_of_features,labelled_image1=label_image(binary_image1)          # Labelled image is required for this analysis
    coincident_image=binary_image1 & binary_image2        # Find pixel overlap between the two images
    coincident_labels=labelled_image1*coincident_image   # This gives a coincident image with the pixels being equal to label
    coinc_list, coinc_pixels = np.unique(coincident_labels, return_counts=True)     # This counts number of unique occureences in the image
    # Now for some statistics
    total_labels=labelled_image1.max()
    total_labels_coinc=len(coinc_list)
    fraction_coinc=total_labels_coinc/total_labels
    
    # Now look at the fraction of overlap in each feature
    # First of all, count the number of unique occurances in original image
    label_list, label_pixels = np.unique(labelled_image1, return_counts=True)
    fract_pixels_overlap=[]
    for i in range(len(coinc_list)):
        overlap_pixels=coinc_pixels[i]
        label=coinc_list[i]
        total_pixels=label_pixels[label]
        fract=1.0*overlap_pixels/total_pixels
        fract_pixels_overlap.append(fract)
    
    
    # Generate the images
    coinc_list[0]=1000000   # First value is zero- don't want to count these. 
    coincident_features_image=np.isin(labelled_image1,coinc_list)   # Generates binary image only from labels in coinc list
    coinc_list[0]=0
    non_coincident_features_image=~np.isin(labelled_image1,coinc_list)  # Generates image only from numbers not in coinc list.
    
    return coinc_list,coinc_pixels,fraction_coinc,coincident_features_image,non_coincident_features_image,fract_pixels_overlap




# Load images

donor=load_image(donor_path)
acceptor=load_image(acceptor_path)
FRET=load_image(FRET_path)



if calculate_autof==1:
    donor_af=donor.mean()
    acceptor_af=acceptor.mean()
    FRET_af=FRET.mean()
    
    print("Donor autofluorescence = %d.\r"%donor_af)
    print("Acceptor autofluorescence = %d.\r"%acceptor_af)
    print("FRET autofluorescence = %d.\r"%FRET_af)

if calculate_bleedthrough==1:
    thresh,donor_binary=threshold_image_otsu(donor)
    
   
    cross_talk=FRET/donor
    
    cross_talk_thesh=cross_talk*donor_binary
    
    cross_talk_list=cross_talk[cross_talk !=0].flatten()
    
    cross_talk_value=cross_talk_thesh.mean()

    print("Crosstalk = %s.\r"%cross_talk_value)
    
if calculate_direct_excitation==1:
    
    acceptor_t=acceptor-autof_acceptor
    thresh,acceptor_binary=threshold_image_otsu(acceptor_t)
    
   
    direct=(FRET-autof_FRET)/(acceptor-autof_acceptor)
    
    direct_thresh=direct*acceptor_binary
    
    imsr2 = Image.fromarray(direct)
    imsr2.save(donor_path+'_direct.tif')
    
    direct_list=direct_thresh[direct_thresh>0].flatten()
    
    direct_value=direct_list.mean()

    print("Direct excitation = %s.\r"%direct_value)
    

if FRET_analysis==1:
    
    donor_corrected=donor-autof_donor
    acceptor_corrected=acceptor-autof_acceptor
    FRET_corrected=FRET-crosstalk*donor-direct_excitation*acceptor-autof_FRET
    

    # Make FRET_image:
    FRET_image=FRET_corrected/(donor_corrected+FRET_corrected)


    # Need to threshold acceptor_corrected image (not affected by FRET)
    thresh,acceptor_binary=threshold_image_otsu(acceptor)

    imsr2 = Image.fromarray(acceptor_binary)
    imsr2.save(acceptor_path+'_binary.tif')

    # Threshold donor:

    thresh,donor_binary=threshold_image_otsu(donor)
    imsr2 = Image.fromarray(donor_binary)
    imsr2.save(donor_path+'_binary.tif')
    
    
    imsr2 = Image.fromarray(donor_corrected)
    imsr2.save(donor_path+'_corrected.tif')
    # Thresholded in both channels
    
    mask=donor_binary*acceptor_binary
    imsr2 = Image.fromarray(mask)
    imsr2.save(root_path+'mask.tif')
    
    FRET_thresholded=mask*FRET_image
    
    imsr2 = Image.fromarray(FRET_thresholded)
    imsr2.save(root_path+'FRET_thresholded.tif')
    
#  Extract all FRET data

    FRET_pixel_list=FRET_thresholded[FRET_thresholded!=0].flatten()
    
# Plot histogram:

    plt.hist(FRET_pixel_list, bins = 20,range=[0,1], rwidth=0.9,color='#ff0000')
    plt.xlabel('FRET Efficiency',size=20)
    plt.ylabel('Number of pixels',size=20)
    plt.savefig(root_path+"Pixel_FRET_hist.pdf")
    plt.show()

#   Feature analysis
    number,labelled=label_image(mask)
    print("%d feautres were detected in the image."%number)
    imsr2 = Image.fromarray(labelled)
    imsr2.save(root_path+'Label.tif')

    measurements=analyse_labelled_image(labelled,FRET_image)

    frets=measurements['mean_intensity']
    plt.hist(frets, bins = 20,range=[0,1], rwidth=0.9,color='#ff0000')
    plt.xlabel('FRET Efficiency',size=20)
    plt.ylabel('Number of Features',size=20)
    plt.savefig(root_path+"_FRET_hist.pdf")
    plt.show()

    areas=measurements['area']
    plt.hist(areas, bins = 30,range=[0,30], rwidth=0.9,color='#ff0000')
    plt.xlabel('Area (pixels)',size=20)
    plt.ylabel('Number of Features',size=20)
    plt.savefig(root_path++"_area_hist.pdf")
    plt.show()

    length=measurements['major_axis_length']
    plt.hist(length, bins = 5,range=[0,10], rwidth=0.9,color='#ff0000')
    plt.xlabel('Length',size=20)
    plt.ylabel('Number of Features',size=20)
    plt.title('Cluster lengths',size=20)
    plt.savefig(root_path+"Lengths.pdf")
    plt.show()





