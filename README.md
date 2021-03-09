# Mandelbrot-Set-Image-Generator


<img src="/set_images/mandelbrot1.jpeg" alt="drawing" width="470"/>


## Requirements

* You must have ***numpy*** in order to convert tensors to images
* You must have an Nvidia graphics card and ***cuda 11.0 or higher*** in order to take advantage of the GPU acceleration
* You must have ***tensorflow gpu*** installed, or in newer models of TF the gpu library is included
* You should install ***numba*** if you are to use GPU acceleration. This is not absolutely necessary, but you may need 
to clear your GPU memory if you use the program to generate multiple large images

## How to use

Once you have all of the basic requirements, you can use the <code>generate_set_image_tf()</code> function to generate a
numpy array that can be converted into an image.

### Parameters:

* <code>x</code> The width of the frame.
* <code>y</code> The height of the frame.
* <code>real_origin</code> The real argument of the origin of the frame.
* <code>imag_origin</code> The imaginary argument of the origin of the frame.
* <code>frame_size</code> Effectively the zoom level of the frame. A smaller value means more zoom. Must be a positive float.
* <code>num_iter</code> The number of times that the main algorithm will iterate: <code>z<sub>n</sub> = z<sub>n-1</sub><sup>2</sup> + c, where z<sub>0</sub> = c</code>
* <code>max_dist</code> Defines which numbers are inside and outside of the set after iteration. If <code>abs(a+bi) <= max_dist</code>, then <code>a+bi</code> is in the set.
* <code>colors</code> Used only in the color version of the algorithm *not yet functional*. Defined by a list of lists of length 3 which contain color triplets from 0-1 in RGB format
