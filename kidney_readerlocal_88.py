## read image and label

import os
import numpy as np
import scipy.misc as misc
import imageio
from skimage import transform as imgtf
import scipy.io as io
import tensorflow as tf
import scipy.ndimage as ndi
import math
from PIL import Image
import random
ceil = math.ceil
hw=425

class Image_Reader(object):
    def __init__(self, image_dir, data_list, input_size, status_index):
        self.image_dir = image_dir
        self.data_list = data_list
        self.status_index = status_index 
        self.image_list, self.label_list = self.read_pred_label_list(self.image_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        print("images"+repr(self.images))
        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=input_size is not None) # Not shuffling if it is val.
        self.image_img, self.mask, self.mask_img, self.c_mask= tf.py_func(self.read_image_from_disk,self.queue,[tf.float32,tf.uint8,tf.float32,tf.float32])
        self.image_img = tf.reshape(self.image_img,[input_size[0],input_size[1],3])
        self.mask_img = tf.reshape(self.mask_img,[input_size[0],input_size[1],1])
        self.mask = tf.reshape(self.mask,[1,1])
        self.c_mask = tf.reshape(self.c_mask,[input_size[0],input_size[1],1])


    def dequeue(self, num_elements):
        image_img_batch, mask_batch, mask_img_batch, cmask_batch= tf.train.batch([self.image_img,self.mask,self.mask_img,self.c_mask],
                                                  num_elements)
        return image_img_batch, mask_batch, mask_img_batch, cmask_batch

## read image and label list function
    def read_pred_label_list(self,image_dir,data_list):
        f = open(data_list, 'r')
        images = []
        masks = []
        segmasks = []
        for line in f:
            image= line.strip("\n")
            images.append(os.path.join(image_dir,image))
            masks.append(image)
        return images, masks   

## image preprocessing
    def crop_kidney(self, input_im):
        output = np.zeros((321,hw))
        index = np.where(input_im>0.0)
        A_r=index[0]
        rmin=np.amin(A_r)
        rmax=np.amax(A_r)
        A_c=index[1]
        cmin=np.amin(A_c)
        cmax=np.amax(A_c)
        output[rmin+20:rmax-5,cmin:cmax+1] = input_im[rmin+20:rmax-5,cmin:cmax+1]
        return output
## crop for data augmentation         
    def randomcrop_kidney(self, input_im):
        padding = 5
        oshape = np.shape(input_im)
        oshape = (oshape[0] + 4*padding, oshape[1] + 2*padding)
        npad = ((2*padding, 2*padding), (padding, padding))
        image_pad = np.lib.pad(input_im, pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - 321)
        nw = random.randint(0, oshape[1] - hw)
        image_crop = image_pad[nh:nh + 321, nw:nw + hw]
        return image_crop    

## rotate for data augmentation        
    def random_rotate(self, input_image):
        im = Image.fromarray(input_image)
        tt = random.randint(0,1)
        if tt==1:
           im_out = im.transpose(Image.FLIP_LEFT_RIGHT)
        else:
           im_out = im
        ii=random.randint(-45,45)
        im_rotate = im_out.rotate(ii)
        return np.array(im_rotate)

## read image and label function    
    def read_image_from_disk(self,img_filename,label_filename):
        c_mask = np.zeros((321,hw,1))
        img3 = np.zeros((321,hw,3)) 
        img_filename = img_filename.decode()
        img = imageio.imread(img_filename)      
        img = img.astype("float32")
        img = imgtf.resize(img,[321,428])       
        img1 = img[:,1:426]
        img0 = self.crop_kidney(img1)
        max_ = np.amax(img0)
        min_ = np.amin(img0)
        img0 = 255.0*(img0 - min_) / (max_ - min_)
        if self.status_index==1: 
           img0 = self.randomcrop_kidney(img0)
           img0 = self.random_rotate(img0)       
        img3[:,:,0]=img0
        img3[:,:,1]=img0
        img3[:,:,2]=img0
        label_image = img0/255.0        
        lab_filename = label_filename.decode()
        if lab_filename[0]=='0':
           label = 0
        else:
           label = 1
           c_mask = c_mask+1 
        label = np.uint8(label)
        return img3.astype("float32"),label.astype("uint8"),label_image.astype("float32"),c_mask.astype("float32")