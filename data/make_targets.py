# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os

from bs4 import BeautifulSoup
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np

XML_PATH = 'imgs/bnd_box/'

def get_img_num(img_file):
    return os.path.basename(img_file).split('.')[0]

def grab_xml_file(img_file):
    num = get_img_num(img_file)
    return XML_PATH + num + '.xml'

def decode_bndbox(xml):
    soup = BeautifulSoup(xml, 'xml')
    boxes = []
    for box in soup.annotation.find_all('bndbox'):
        boxes.append((
            int(box.xmin.contents[0]),
            int(box.xmax.contents[0]),
            int(box.ymin.contents[0]),
            int(box.ymax.contents[0])
        ))
    return boxes

def make_target(img_file, boxes):
    img = Image.open(img_file)
    img_array = image.img_to_array(img, data_format='channels_last')
    shape = img_array.shape
    target = np.zeros(shape)
    for box in boxes:
        xmin, xmax, ymin, ymax = box
        target[ymin:ymax, xmin:xmax, :] = 1
    return target