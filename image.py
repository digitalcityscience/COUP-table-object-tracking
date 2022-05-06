from typing import Any, List

import cv2
import numpy

convolution_kernel = numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

points_src = numpy.array([[0, 0], [0, 800], [1280, 0], [1280, 800]])
points_dst = numpy.array([[0, 0], [0, 1000], [1000, 0], [1000, 1000]])


def buffer_to_array(image_buffer) -> List:
    return numpy.asanyarray(image_buffer)


def read_from_file(file_name: str):
    return cv2.imread(filename=file_name)

def write_to_file(buffer, file_name:str = "output.png"):
  cv2.imwrite(f'./output/{file_name}', numpy.asanyarray(buffer))

def sharpen_and_rotate_image(image: List) -> List:
    #grayscale_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    grayscale_image = image
    sharpened_image = cv2.filter2D(grayscale_image, -1, convolution_kernel)
    transformation, _ = cv2.findHomography(points_src, points_dst)
    return cv2.warpPerspective(sharpened_image, transformation, (1000, 1000))
