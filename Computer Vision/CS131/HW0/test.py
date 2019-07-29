from __future__ import print_function
import random
import numpy as np;
from linalg import *
from imageManip import *
import matplotlib.pyplot as plt

M=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]);
a=np.array([1,1,0]);
b=np.array([[-1],[2],[5]]);

print("M = \n", M)
print("a = ", a)
print("b = ", b)

aDotB = dot_product(a, b)
print (aDotB);

ans = matrix_mult(M, a, b)
print (ans)

print(get_singular_values(M, 1))
print(get_singular_values(M, 2))

M = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
val, vec = get_eigen_values_and_vectors(M[:,:3], 1)
print("Values = \n", val)
print("Vectors = \n", vec)
val, vec = get_eigen_values_and_vectors(M[:,:3], 2)
print("Values = \n", val)
print("Vectors = \n", vec)


image1_path = './image1.jpg'
image2_path = './image2.jpg'

def display(img):
    # Show image
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
image1 = load(image1_path)
image2 = load(image2_path)

display(image1)
display(image2)

new_image = change_value(image1)
display(new_image)

grey_image = convert_to_grey_scale(image1)
display(grey_image)

without_red = rgb_decomposition(image1, 'R')
without_blue = rgb_decomposition(image1, 'B')
without_green = rgb_decomposition(image1, 'G')

display(without_red)
display(without_blue)
display(without_green)


image_l = lab_decomposition(image1, 'L')
image_a = lab_decomposition(image1, 'A')
image_b = lab_decomposition(image1, 'B')

display(image_l)
display(image_a)
display(image_b)

image_h = hsv_decomposition(image1, 'H')
image_s = hsv_decomposition(image1, 'S')
image_v = hsv_decomposition(image1, 'V')

display(image_h)
display(image_s)
display(image_v)

image_mixed = mix_images(image1, image2, channel1='R', channel2='G')
display(image_mixed)





