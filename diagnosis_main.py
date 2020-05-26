"""Training instance-level classification model on the kidney dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
from absl import app as absl_app
from absl import flags
import tensorflow as tf  # 
from deeplab_model import DeepLabLFOVModel
from kidney_readerlocal_88 import Image_Reader
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
import numpy as np
import time
import scipy.io as sio

### Training parameters setting
_NUM_CLASSES = 2
batch_size = 5
INPUT_SIZE = (321,425)
Base_Rate = 1e-5
NUM_STEPS=100001

## path setting
DATASET_NAME = 'Kidney'
TRAIN_LIST_PATH = 'train_imageS.txt' #import train data list
TEST_LIST_PATH = 'test_imageS.txt' #import validation data list
IMAGE_PATH = './dataset/' ## image path 
model_save = './model/' #model path 
snapshot_dir = model_save+'snapshots/'
pretrain_dir = 'VGG_16.npy'## pretrain model path

     
def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
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
      pretrain_dir: The save dir of the pretrained VGG16 model
      fullsize: The output feature size of last conv layer
      deeplabmodel: The classification model
    """
    self.num_classes =  _NUM_CLASSES
    self.model_weights = pretrain_dir
    self.final_size = fullsize
    self.deeplab_model = deeplabmodel
 
  ## The last full connected layer for classification
  def classify_layer(self, input_feature,is_training):
      input_feature = tf.reshape(input_feature,[-1,self.final_size])
      fc_weight0 = tf.get_variable("first_fc_w",shape=[self.final_size,64*4])
      fc_bias0 = tf.get_variable("first_fc_b",shape=[64*4])
      output = tf.matmul(input_feature,fc_weight0)+fc_bias0
      output = tf.layers.batch_normalization(output,training=is_training)
      output = tf.nn.relu(output)
      output = tf.cond(is_training,lambda:tf.nn.dropout(output,0.5),lambda:output)
      fc_weight1 = tf.get_variable("final_fc_w",shape=[64*4,self.num_classes])
      fc_bias1 = tf.get_variable("final_fc_b",shape=[self.num_classes])
      output = tf.matmul(output,fc_weight1)+fc_bias1


      pred_one_hot = tf.one_hot(tf.argmax(output,axis=1),self.num_classes)
      output_one_hot = tf.reduce_mean(output*pred_one_hot)
    ## compute corresponding cam
      gradients = None
      gradients = tf.gradients(output_one_hot,input_feature)[0]
      gradients = tf.div(gradients,tf.sqrt(tf.reduce_mean(tf.square(gradients)))+1e-5)
      gradients = tf.reshape(gradients,[-1,41,54,512])
      gradients = tf.reduce_mean(gradients,axis=(1,2),keepdims=True)

      cams = None
      for i in range(batch_size):
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

      return output, cams_normalized
  
  ## Output classification probability and cam     
  def multi_model(self,input_image,is_training): 
      _,_,feature512,_ = self.deeplab_model.preds(input_image)
      out_probability, cams_probability = self.classify_layer(feature512,is_training)
      return out_probability, cams_probability

 


   
    

def main():
##########################################################
# DATA reader
# read train and test pred and label images
##########################################################
    coord = tf.train.Coordinator()
    trainreader = Image_Reader(
                 IMAGE_PATH,
           TRAIN_LIST_PATH,
              INPUT_SIZE,
              1,
              )
    trainimage_batch, trainmask_batch, trainmaskimg_batch, train_cmask_batch = trainreader.dequeue(batch_size)
    
    testreader = Image_Reader(
                     IMAGE_PATH,
                 TEST_LIST_PATH,
                 INPUT_SIZE,
                 0,
                 )
    testimage_batch, testmask_batch, testmaskimg_batch, test_cmask_batch= testreader.dequeue(batch_size)
    netindex = tf.placeholder(tf.bool)
    image_batch = tf.cond(netindex, lambda: trainimage_batch, lambda: testimage_batch)
    label_batch = tf.cond(netindex, lambda: trainmask_batch, lambda: testmask_batch)#label classes
    mask_batch = tf.cond(netindex, lambda: trainmaskimg_batch, lambda: testmaskimg_batch)
    cmask_batch = tf.cond(netindex, lambda: train_cmask_batch, lambda: test_cmask_batch)
    label_batch = tf.squeeze(tf.squeeze(label_batch,axis=1),axis=1)
    print("label_batch"+repr(label_batch))
##########################################################
#   bulid network
##########################################################
    net_deeplab = DeepLabLFOVModel(pretrain_dir) 
    kidneymodel = KidneyclassificationModel(_NUM_CLASSES,pretrain_dir,41*54*512,net_deeplab)
    logits,cams = kidneymodel.multi_model(image_batch,netindex) #probability
    print("image_batch"+repr(image_batch))
    prediction = tf.argmax(logits, axis=1) #classes
    print("prediction"+repr(prediction))
    print("logits"+repr(logits))
    label_class = tf.one_hot(label_batch,2)
    print("label_batch"+repr(label_batch))   
    cross_loss = tf.losses.softmax_cross_entropy(onehot_labels=label_class,logits=logits)
    print("cross_loss"+repr(cross_loss))
    pred_accuracy = tf.reduce_sum(tf.cast(tf.equal(prediction,tf.cast(label_batch,tf.int64)),tf.float32))/batch_size
    mask_loss = net_deeplab.loss(image_batch,tf.cast(mask_batch,tf.uint8))
####################################
##### output cam
####################################   
    from heatmap import heatmap
    def to_heatmap(a):
        a = (255*a).astype(np.uint8)
        return 255*heatmap(a).astype(np.float32)
    cams = tf.image.resize_bilinear(tf.expand_dims(cams,axis=3),INPUT_SIZE)
    cams_rgb,mask_rgb,cams_mask_rgb = [],[],[]
    for i in range(batch_size):
        cams_rgb.append(tf.py_func(to_heatmap,[cams[i,:,:,0]],tf.float32))
        mask_rgb.append(tf.py_func(to_heatmap,[cmask_batch[i,:,:,0]*mask_batch[i,:,:,0]],tf.float32))
        cams_mask_rgb.append(tf.py_func(to_heatmap,[cams[i,:,:,0]*mask_batch[i,:,:,0]],tf.float32))
    cams_rgb = tf.stack(cams_rgb,axis=0)
    mask_rgb = tf.stack(mask_rgb,axis=0)
    cams_mask_rgb = tf.stack(cams_mask_rgb,axis=0)
    image1 = tf.concat([image_batch,cams_rgb],axis=2)
    image2 = tf.concat([tf.cast(mask_rgb,tf.float32),cams_mask_rgb],axis=2)
    images = tf.concat([image1,image2],axis=1)
    image_summary = tf.compat.v1.summary.image("image",images,max_outputs=10)
    summary_writer = tf.compat.v1.summary.FileWriter(model_save+"log")
##########################################################
# running the model
##########################################################    
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    loss_old = tf.placeholder(dtype=tf.float32, shape=())
    loss_sum= loss_old+1.0*cross_loss                               
    Learning_Rate = Base_Rate
    optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=Learning_Rate)
    trainable = tf.compat.v1.trainable_variables()
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        optim = optimiser.minimize(loss_sum, var_list=trainable)
    # Set up tf session and initialize variables. 
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
    saver = tf.compat.v1.train.Saver(var_list=var_list,max_to_keep=40)
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    if not os.path.exists(model_save):
           os.makedirs(model_save)

    for step in range(NUM_STEPS):
        start_time = time.time()
        feed_dict0 = { step_ph: step, loss_old: 0, netindex: 1}
        oldloss_train = 0.0
        oldacc_train = 0.0
        oldacc_masktrain = 0.0
        oldloss_crosstrain = 0.0
        oldloss_masktrain = 0.0
        for cstep in range(3):
            feed_dict0 = { step_ph: step, loss_old: 0,netindex: 1}
            loss_train,loss_crosstrain,loss_masktrain,acc_train,acc_masktrain,trainpredict,trainlogit = sess.run([loss_sum,cross_loss,mask_loss,pred_accuracy,dice_M,prediction,logits], feed_dict=feed_dict0)
            oldloss_train = oldloss_train+loss_train 
            oldloss_crosstrain = oldloss_crosstrain+loss_crosstrain
            oldloss_masktrain = oldloss_masktrain + loss_masktrain
            oldacc_train = oldacc_train + acc_train
            oldacc_masktrain = oldacc_masktrain + acc_masktrain            
        feed_dict1 = { step_ph: step, loss_old: oldloss_train,netindex: 1}
        _, loss_trainsum,loss_crosstrain,loss_masktrain,acc_train,acc_masktrain,trainsummarys  = sess.run([optim,loss_sum,cross_loss,mask_loss,pred_accuracy,dice_M,image_summary], feed_dict=feed_dict1)
        oldacc_train = oldacc_train + acc_train
        oldacc_masktrain = oldacc_masktrain + acc_masktrain
        oldloss_crosstrain = oldloss_crosstrain+loss_crosstrain
        oldloss_masktrain = oldloss_masktrain + loss_masktrain
        duration = time.time() - start_time
        if step % 10 == 0:
              print('step {:d} \t trainloss = {:.3f}, ({:.3f} sec/step),trainacc = {:.3f},train_maskacc = {:.3f},train_crossloss = {:.3f},train_maskloss = {:.3f}'.format(step, loss_trainsum, duration,oldacc_train/4.0,oldacc_masktrain/4.0,oldloss_crosstrain,oldloss_masktrain))
              summary_writer.add_summary(trainsummarys,global_step=step)
        if step % 50 == 0:
              oldloss_test = 0.0
              oldacc_test = 0.0
              oldacc_masktest = 0.0
              feed_dict0 = { step_ph: step, loss_old: 0, netindex:0}
              for cstep in range(40):
                  loss_test,loss_crosstest,loss_masktest,acc_test,acc_masktest,testpredict,testlogit,testsummarys = sess.run([loss_sum,cross_loss,mask_loss,pred_accuracy,dice_M,prediction,logits,image_summary], feed_dict=feed_dict0)
                  oldloss_test = oldloss_test+loss_test
                  oldacc_test = oldacc_test + acc_test
                  oldacc_masktest = oldacc_masktest + acc_masktest
              summary_writer.add_summary(testsummarys,global_step=step)
              print('step {:d} \t testloss = {:.3f}, ({:.3f} sec/step),testacc = {:.3f}, test_maskacc = {:.3f}'.format(step, oldloss_test, duration, oldacc_test/40.0,oldacc_masktest/40.0))
        if step % 10000== 0 and step >= 1000:
              save(saver, sess, snapshot_dir, step)
               
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
