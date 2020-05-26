"""Estimating instance-level classification scores"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order
from deeplab_model import DeepLabLFOVModel
import imageio
from skimage import transform as imgtf
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
import numpy as np
import time
import scipy.io as io
import scipy.misc as misc

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 2
hw=425

batch_size = 852 ##The number of slice



_NUM_IMAGES = {
    'train': 5,
    'validation': 5,
}




_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000
INPUT_SIZE = (321,hw)
input_size = (321,hw)
IMAGE_PATH = ''
LIST_PATH = ''
model_weights = ''
pretrain_dir = '' ### the path of pretrained VGG16 model
pred_accuracy_matrix = np.zeros((batch_size,1))
logit_matrix = np.zeros((batch_size,2))
cam_path = './cam/'+str(times)+'times/local_fullimage_256'+str(index)+'_88/'+str(tname)+'/'+str(index2)+'/'


# define some function

if not os.path.exists(matrix_path):
    os.makedirs(matrix_path)

if not os.path.exists(cam_path+'50000/'):
    os.makedirs(cam_path+'50000/')

if not os.path.exists(feature_path+'50000/'):
    os.makedirs(feature_path+'50000/')
        
def dice_mask(input_batch1, input_batch2):
    input_1=tf.equal(input_batch1, 1.0, name=None)
    print ("input_1:"+repr(input_1))
    input_2=tf.equal(input_batch2, 1.0, name= None)
    print ("input_2:"+repr(input_2))
    accuracy_n=tf.logical_and(input_1, input_2,name=None)
    input2_z=tf.cast(input_2, tf.int32)
    input1_z=tf.cast(input_1, tf.int32)
    accuracy_z=tf.cast(accuracy_n, tf.int32)
    accuracy_n= 2*tf.reduce_sum(accuracy_z)/(tf.reduce_sum(input1_z)+tf.reduce_sum(input2_z))
    print ("accuracy_mask:"+repr(accuracy_n))
    return accuracy_n
    
# define read image function
def image_slice(image_batchs, index):
    image_s = image_batchs[index,:,:,:]
    return image_s  

def index_slice(index_batchs, index):
    index_s = index_batchs[index,:]
    return index_s 
        
def read_pred_label_list(image_dir,data_list):
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        image= line.strip("\n")
        images.append(os.path.join(image_dir,image))
        masks.append(image)
    return images, masks  

def crop_kidney(input_im):
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


def read_image_from_disk(img_filename_batch,label_filename_batch,ii):
    label_filename = label_filename_batch[ii]
    img_filename = img_filename_batch[ii]
    img3 = np.zeros((321,hw,3))
    c_mask = np.zeros((321,hw,1))
    img = imageio.imread(img_filename)      
    img = img.astype("float32")
    img = imgtf.resize(img,[321,428]) 
    img1=  img[:,1:426]
    img0=img1
    img0 = crop_kidney(img1)
    max_ = np.amax(img0)
    min_ = np.amin(img0)
    img0 = 255.0*(img0 - min_) / (max_ - min_)
    img3[:,:,0]=img0
    img3[:,:,1]=img0
    img3[:,:,2]=img0
    label_image = img0/255.0        
    if label_filename[1]=='o' or label_filename[0]=='0':
       label = 0
    else:
       label = 1
       c_mask = c_mask+1
    label = np.uint8(label)
    return img3.astype("float32"),label.astype("uint8"),label_image.astype("float32"),c_mask.astype("float32") 
    
##########################################################
# construct the model
##########################################################
class KidneyclassificationModel():
  """Model class with appropriate defaults for Imagenet data."""
  def __init__(self,_NUM_CLASSES, pretrain_dir,fullsize,deeplabmodel):
    """These are the parameters that work for Imagenet data.
    Args:
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      model_weights: The save dir of the pretrained VGG16 model
    """
    self.num_classes =  _NUM_CLASSES
    self.final_size = fullsize
    self.deeplab_model = deeplabmodel
 
  
  def classify_layer(self, input_feature,is_training):
      input_feature = tf.reshape(input_feature,[-1,self.final_size])
      fc_weight0 = tf.get_variable("first_fc_w",shape=[self.final_size,64*4])
      fc_bias0 = tf.get_variable("first_fc_b",shape=[64*4])
      output = tf.matmul(input_feature,fc_weight0)+fc_bias0
      output = tf.layers.batch_normalization(output,training=is_training)
      output = tf.nn.relu(output)
      output = tf.cond(is_training,lambda:tf.nn.dropout(output,0.5),lambda:output)
      output_feature = output
      fc_weight1 = tf.get_variable("final_fc_w",shape=[64*4,self.num_classes])
      fc_bias1 = tf.get_variable("final_fc_b",shape=[self.num_classes])
      output = tf.matmul(output,fc_weight1)+fc_bias1


      pred_one_hot = tf.one_hot(tf.argmax(output,axis=1),self.num_classes)
      output_one_hot = tf.reduce_mean(output*pred_one_hot)

      gradients = None
      gradients = tf.gradients(output_one_hot,input_feature)[0]
      gradients = tf.div(gradients,tf.sqrt(tf.reduce_mean(tf.square(gradients)))+1e-5)
      gradients = tf.reshape(gradients,[-1,41,54,512])
      gradients = tf.reduce_mean(gradients,axis=(1,2),keepdims=True)

      cams = None
      for i in range(1):
          tmp = tf.reshape(input_feature[i],[1,41,54,512])
          single_cam = tmp*gradients[i]
          single_cam = tf.reduce_sum(single_cam,axis=3)
          single_cam = tf.nn.relu(single_cam)
          if cams is None:
              cams = single_cam
          else:
              cams = tf.concat([cams,single_cam],axis=0)

      cams_max = tf.reduce_max(cams,[1,2],keepdims=True)
      cams_normalized = cams/(cams_max+1e-8)      

      return output, cams_normalized,output_feature,fc_weight1,fc_bias1
  
      
  def multi_model(self,input_image,is_training): 
      deeplab_mask,_,feature512,_ = self.deeplab_model.preds(input_image)
      out_probability, cams_probability, output_feature,ww1, bb1 = self.classify_layer(feature512,is_training)
      return out_probability, deeplab_mask, cams_probability, output_feature,ww1,bb1
  
    
def main():
    # read train and test pred and label images
    images_namesbatch,masks_namesbatch = read_pred_label_list(IMAGE_PATH,LIST_PATH)
    kidneymask_batch = np.zeros((batch_size,input_size[0],input_size[1],3))
    kidneyc_batch = np.zeros((batch_size,1))
    labelmask_batch = np.zeros((batch_size,input_size[0],input_size[1]))
    cmask_batch = np.zeros((batch_size,input_size[0],input_size[1],1)) 
    for tt in range(batch_size):
         kidneymask_batch[tt,:,:,:],kidneyc_batch[tt,:],labelmask_batch[tt,:,:], cmask_batch[tt,:,:,:]= read_image_from_disk(images_namesbatch,masks_namesbatch,tt)
    kidneymask_batch = np.reshape(kidneymask_batch,(batch_size, input_size[0],input_size[1], 3))
    kidneyc_batch = np.reshape(kidneyc_batch,(batch_size, 1))
    labelmask_batch = np.reshape(labelmask_batch,(batch_size, input_size[0],input_size[1], 1))
    cmask_batch = np.reshape(cmask_batch,(batch_size, input_size[0],input_size[1], 1))
    kidneymask_batch = tf.convert_to_tensor(kidneymask_batch, dtype=tf.float32) 
    kidneyc_batch = tf.convert_to_tensor(kidneyc_batch, dtype=tf.uint8) 
    labelmask_batch = tf.convert_to_tensor(labelmask_batch, dtype=tf.float32)
    cmask_batch = tf.convert_to_tensor(cmask_batch, dtype=tf.float32)
    ind = tf.placeholder(tf.int32, shape=(1, 1)) 
    kidneymask_slice =  tf.py_func(image_slice, [kidneymask_batch,ind], tf.float32)
    kidneyc_slice =  tf.py_func(index_slice, [kidneyc_batch,ind], tf.uint8)
    labelmask_slice = tf.py_func(image_slice, [labelmask_batch,ind], tf.float32) 
    cmask_slice = tf.py_func(image_slice, [cmask_batch,ind], tf.float32) 
    ###
    kidneymask_slice = tf.reshape(kidneymask_slice,[1,input_size[0],input_size[1],3])
    labelmask_slice = tf.reshape(labelmask_slice,[1,input_size[0],input_size[1],1])
    kidneyc_slice = tf.reshape(kidneyc_slice,[1,1])
    cmask_slice = tf.reshape(cmask_slice,[1,input_size[0],input_size[1],1])
    # bulid network
##########################################################
# input the images and output results
##########################################################
    net_deeplab = DeepLabLFOVModel(pretrain_dir) 
#    net_regression = RegressionNet()
    kidneymodel = KidneyclassificationModel(_NUM_CLASSES,pretrain_dir,41*54*512,net_deeplab)
    netindex = tf.placeholder(tf.bool)
    logits,mask,cams,feature,ww0,bb0 = kidneymodel.multi_model(kidneymask_slice,netindex)
    print("logits"+repr(logits)) #probability
    print("kidneymask_slice"+repr(kidneymask_slice))
    prediction = tf.argmax(logits, axis=1) #classes
    print("prediction"+repr(prediction))
    print("logits"+repr(logits))
    pred_accuracy = tf.reduce_sum(tf.cast(tf.equal(prediction,tf.cast(kidneyc_slice,tf.int64)),tf.float32))
    dice_M = dice_mask(tf.cast(mask,tf.float32), labelmask_slice)
    # parameter setting
    # network setting
    trainable = tf.trainable_variables()
    print("trainable"+repr(trainable))                       
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)    

    g_list = tf.global_variables()
    batch_norm_variables = [g for g in g_list if "moving_" in g.name]
    print("batch_norm_variables:%s" % str(batch_norm_variables))
    var_list = []
    var_list.extend(trainable)
    var_list.extend(batch_norm_variables)
    saver = tf.train.Saver(var_list=var_list,max_to_keep=20)
    saver.restore(sess, model_weights) 

    start_time = time.time()
#    print(sess.run(ww0), sess.run(bb0))
    for ss in range(batch_size):
        accuracy,logit,cams_out,feature_out = sess.run([pred_accuracy,logits,cams,feature],feed_dict={ind: np.reshape(ss,(1,1)),netindex:0}) # train:1 test:0
        mname = masks_namesbatch[ss]
        pred_accuracy_matrix[ss,:]= accuracy
        logit_matrix[ss,:]=logit
    print("pred_accuracy_matrix"+repr(np.mean(pred_accuracy_matrix)))

    
        
    

              
    
if __name__ == '__main__':
    main()    
    
