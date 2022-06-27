from typing import List

import cv2
import numpy

def buffer_to_array(image_buffer) -> List:
    return numpy.asanyarray(image_buffer)


def read_from_file(file_name: str):
    return cv2.imread(filename=file_name)

def write_to_file(buffer, file_name:str = "output.png"):
  cv2.imwrite(f'./output/{file_name}', numpy.asanyarray(buffer))

def sharpen_and_rotate_image(grayscale_image: List) -> List:
    convolution_kernel = numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(grayscale_image, -1, convolution_kernel)
    return sharpened_image
