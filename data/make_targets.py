# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os
import glob

from bs4 import BeautifulSoup
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np
import imageio

from utils.params import *

def get_img_num(img_file):
    return os.path.basename(img_file).split('.')[0]

def grab_xml_file(img_file):
    i = get_img_num(img_file)
    return f'{XML_PATH}{i}.xml'

def decode_bndbox(xml):
    soup = BeautifulSoup(xml, 'html.parser')
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

if __name__ == '__main__':
    img_files = glob.glob(f'{IMG_PATH}*.jpg')
    if not os.path.exists(TRG_PATH):
        os.mkdir(TRG_PATH)
    for img in img_files:
        xml_file = grab_xml_file(img)
        i = get_img_num(xml_file)
        raw_xml = open(xml_file, 'r')
        boxes = decode_bndbox(raw_xml)
        target = make_target(img, boxes)
        imageio.imwrite(f'{TRG_PATH}{i}.png', target)
