import numpy as np
import glob
from PIL import Image
import os

def jpg_size(self,jpg_dir,out_dir,class_name,width=100,height=100):
    for jpg_file in glob.glob(jpg_dir + r'\*.jpg'):
        self.convert_size(jpg_file, out_dir, class_name, width, height)

def convert_size(self, jpg_file, out_dir, class_name, width, height):
    img = Image.open(jpg_file)
    new_img = img.resize((width,height),Image.BILINEAR)
    new_img.save(os.path.join(out_dir,class_name + '_' + os.path.basename(jpg_file) + '.jpg'))

def jpg2array(self, jpg_dir, out_file, width=100,height=100, jpg_num=1, jpg_type=3):
