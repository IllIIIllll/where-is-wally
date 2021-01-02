# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os

from bs4 import BeautifulSoup

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