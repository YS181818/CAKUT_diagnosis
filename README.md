Multi-instance deep learning of ultrasound imaging data for pattern classification of congenital abnormalities of the kidney and urinary tract in children-TensorFlow
===
This is an implementation of the proposed model in TensorFlow for instance-level classification of congenital abnormalities of the kidney and urinary tract (CAKUT).

Model Description：
------
We adopt the transfer learning to finetune a deep learning classification network based on a pre-trained VGG16 model for learning discriminative features from individual 2D ultrasound images and estimating instance-level classification scores.<br>
Then, a mean pooling operator is adopted to fuse the multiple instance-level classification scores for generating an overall (bag-level) classification score for each subject.


For more details on the underlying model please refer to the following paper:
-------
@article{yin2020Multiinstance,<br>
title={Multi-instance deep learning of ultrasound imaging data for pattern classification of congenital abnormalities of the kidney and urinary tract in children},<br>
author={Shi Yin, Qinmu Peng, Hongming Li, Zhengqiang Zhang, Xinge You, Katherine Fischer, Susan L. Furth, Yong Fan, Gregory E. Tasian},<br>
journal={Urology},<br>
year={2020}}<br>

Requirements：
--------
The proposed networks were implemented based on Python 3.7.0 and TensorFlow r1.11


Training：
--------
We initialized the network from the VGG_16.npy<br>
Training the instance-level classification: diagnosis_main.py


Evaluation:
----------
Computing the instance-level classification score: compute_classification.py




