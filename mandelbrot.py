#!/usr/bin/env python


import numpy as np
import math
from PIL import Image

def in_set(c, num_iter, max_dist):
    z = c
    for i in range(num_iter):
        z = (z*z) + c
    if np.abs(z) <= max_dist:
        return 1
    else:
        return 0

def create_image_array(x, y, real_origin, imag_origin, frame_size, num_iter, max_dist):
    arr = np.zeros((x, y))
    
    ratio = y/x
    
    complex_multiplier = 1j
    for i in range(x):
        for j in range(y):
            t1 = ((i+1)/x)*frame_size - (frame_size/2) + real_origin
            t2 = (((((j+1)/y)*frame_size - (frame_size/2))*ratio) + imag_origin)*complex_multiplier
            arr[i,j] = in_set(t1+t2, num_iter, max_dist)
    return arr.T



def in_set_color(c, num_iter, max_dist):
    z = c
    i = 0
    while i < num_iter:
        z = (z*z) + c
        if np.abs(z) >= max_dist:
            break
        i += 1
    if i > num_iter/2:
        return i/num_iter
    else:
        return 0
    

    
def create_image_array_color(x, y, real_origin, imag_origin, frame_size, num_iter, max_dist):
    arr = np.zeros((y, x,3))
    
    ratio = y/x
    
    complex_multiplier = 1j
    for i in range(x):
        for j in range(y):
            t1 = ((i+1)/x)*frame_size - (frame_size/2) + real_origin
            t2 = (((((j+1)/y)*frame_size - (frame_size/2))*ratio) + imag_origin)*complex_multiplier
            color = in_set_color(t1+t2, num_iter, max_dist)
            arr[j,i,1] = 1-color
            arr[j,i,2] = color
            
    return arr





width = 1080
height = 1080
real_origin = 0.35
imag_origin = 0
frame_size = 0.3
num_iter = 40
max_dist = 4






set_arr = create_image_array_color(width, height, real_origin, imag_origin, frame_size, num_iter, max_dist)



picture = Image.fromarray(set_arr.astype(np.uint8)*255)
picture.save('mandelbrot.jpeg')