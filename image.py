from typing import List
import numpy
import cv2


def buffer_to_array(image_buffer) -> List:
  return numpy.asanyarray(image_buffer)

def read_from_file(file_name:str):
  return cv2.imread(filename=file_name)
