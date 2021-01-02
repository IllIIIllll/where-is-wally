# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os

XML_PATH = 'imgs/bnd_box/'

def get_img_num(img_file):
    return os.path.basename(img_file).split('.')[0]

def grab_xml_file(img_file):
    num = get_img_num(img_file)
    return XML_PATH + num + '.xml'