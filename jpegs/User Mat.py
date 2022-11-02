# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:05:11 2021

@author: n.naik
"""

import numpy as np
from skimage import io
import cv2 
import glob
import scipy.io

no = int(input("Enter number of Images: "))         #Number of images for Bscans
ang = int(360 / no)         #Angle step between consecutive images
x = int(0)          #Default angle value that can be increased by angle step
#x = 0


m = int(input("Enter Width of Image in pixels: "))      #No. of pixels in Width
n = int(input("Enter Height of Image in pixels: "))         #No. of pixels in Height

img = np.empty(shape=(n, m), dtype = np.float64)    #Empty array of shape NxM - same as our image
ii = np.empty(shape=(n, m), dtype=np.float32)       #Another Empty array of shape NxM - same as our image
Bscans = np.empty(shape=(no, n, m), dtype=np.float32)   #Array where our all images would be saved


Bscan_dims = np.array([n, m], dtype = np.uint16)        #Saving dimentions of images
Bscan_dims.shape = (1,2)        #Shape of array i.e used to save the dimentions of images 
print(Bscan_dims)


path = input("Enter path to read the Images: ")         #Input the path of all Bscans NOTE: No additional images in this folder
filelist = glob.glob(path + '/*.jpg')       #File list where all images are read and saved
abc  = filelist         #creating same file list for future use
hij = filelist      #Another file list for same as bscans list
blackwhite = filelist       #Same file list hij to save black and white version of the given images
#img[x] = io.imread('s0.jpg', as_gray=True)
#i[x] = np.array(img[x], dtype=np.float32)


for i in range(no):         #Loop to convert each images to B&W and save them to defined filelist
    x = i * ang
    x1 = str
    x1 = x
    hij[i] = io.imread( abc[i], as_gray=True)       #Reading each images one by one converting them to b&w
    blackwhite[i]  = np.array(hij[i], dtype=np.float32)         #Saving each b&w images to a filelist
    #Bscans = np.stack([ii])
    print(x1)

Bscans = np.array(blackwhite)       #Convert filelist to numpy array of all the bscans


length = float(input("Enter Lenth of sample: "))        #Length of sample to be saved in array
L = np.array([length], dtype = np.float32)
L.shape = (1,1)


basename = np.array(['spermatic_duct_'])        #Just a additional text in mat file (No need to change)


center = np.array([0.495204, 0.32814 ], dtype = np.float32)         #Saving the center in mat file
center.shape = (1,2)


minpath = input("Enter array file path for 'Mindist.npy' (Path with file name): ")      #This is same for all the mat file(until special requirements) so importing saved file 
mindists = np.load(minpath)


rad = float(input("Enter radii: "))         #Radius of the capilary used
radii = np.array([3.223553299903869629e-01 ], dtype = np.float32)   #If varying for all the samples replace '3.223..e-01' with 'rad'
radii.shape = (1,1)


tweaks = np.zeros(shape=(no, 2), dtype=np.float32)      #Min and max colour depection for RI, generally same for all
for j in range(no):         #Loop for arranging values for tweaks
    tweaks[j][1] = -0.1125
    print(j)


pathwin = input("Enter array file path for 'Win.npy' (Path with file name): ")  #Window to show our RI which could be changed under special conditions, but loading saved .npy file if same
window = np.load(pathwin)


weightimg = input("Enter Path for Weighted Image (Path with file name): ")  #Loading image to remove white line in the middle of all the bscans
img = io.imread(weightimg, as_gray=True)     #Converting image to b&w
#print(img.shape)
imgweig = cv2.resize(img, (m , n))      #Resizing the image to mathch with our image shape
weights = imgweig[np.newaxis, :, :]	        #Entering the image to array in this format to able to read by OCRT code


x1 = np.zeros([no, m], dtype=np.float32 )       #Array filled with zeros of shape no x m
x2 = np.zeros([no, 0 ], dtype=np.float32 )      #Array filled with zeros of shape no x 0

lup = int (m -1)        #Entering values from 0 to 1 in m
rang = 1 / m        #Finding out a value that can be incremented m times to achive 1 on last step of m

for x in range(lup):     #Loop to save this value after incrementing by steps 
    z = rang
    y = z*x         #Multiplying the value m times but in steps
    x2 = np.insert(x2, [x] , [[y]] , axis=1 )  #Replacing this values on zeros
    print(x, '= ', y)

x2 = np.insert(x2, [lup] , [[1]] , axis=1 ) 

xzcoords = np.stack([x1, x2], axis=2)       #Saving x1 and x2 as numpy array

savepath = input ("Enter Path to save MAT file:")
name = input ("Enter Name for MAT file(Including .mat extension):")
scipy.io.savemat(savepath + '/' + name , {"Bscans":Bscans , "Bscan_dims":Bscan_dims , "L": L, "basename":basename, "center":center, "mindists":mindists, "radii":radii, "tweaks":tweaks, "weights":weights, "window":window, "xzcoords":xzcoords})




