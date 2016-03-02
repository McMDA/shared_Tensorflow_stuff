# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the Tabor network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the "Tabor_FeedForward_Only-2-6-16.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.python.platform
import tensorflow as tf
import numpy as np

# The TABOR1 dataset has 3 classes, a b c.
NUM_CLASSES = 3

# The TABOR1 sentences are always 6 words long
#IMAGE_SIZE = 28
#IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(inputs, batch_size, num_h1_weights):
  """Build the TABOR model up to where it may be used for inference.
  
  This should have the initial weights specified by our dear Tabor

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    (removed for now) hidden2_units: Size of the second hidden layer.

  Returns:
    (outputs from the linear layer) softmax_linear: Output tensor with the computed logits.
  """
  
  # Linear Layer
  with tf.name_scope('hidden1'):
	b1=tf.ones([batch_size,1],name="b1")
	tf.histogram_summary("b1",b1) 
	z1=LinearLayer(inputs,batch_size,num_h1_weights,"w1","Neuron4")
	tf.histogram_summary("z1_Linear",z1)
	z2=LinearLayer(inputs,batch_size,num_h1_weights,"w2","Neuron5")
	tf.histogram_summary("z2_Linear",z1)
	hidden1=tf.concat(1,[b1,z1,z2])
	#Gaussian Layer
  with tf.name_scope('hidden2'):
    b2=tf.zeros([batch_size,1],name="b2")
    tf.histogram_summary("b2",b2)
    z1=GaussLayer(hidden1,b2,batch_size,3,"sigma1","mu1","Neuron6")
    tf.histogram_summary("z1_Gauss",z1)
    z2=GaussLayer(hidden1,b2,batch_size,3,"sigma2","mu2","Neuron7")
    tf.histogram_summary("z2_Gauss",z2)
    z3=GaussLayer(hidden1,b2,batch_size,3,"sigma3","mu3","Neuron8")
    tf.histogram_summary("z3_Gauss",z3)
    hidden2=tf.concat(1,[b2,z1,z2,z3])
  with tf.name_scope('output'):  
    output=SoftmaxLayer(hidden2,4) 
    tf.histogram_summary("z_out",output)   
  return output

  
def LinearLayer(X,batch_size,num_h1_weights,weightName,nameScope):
  with tf.name_scope(nameScope):
    w1= tf.Variable(tf.truncated_normal(shape=[num_h1_weights,1],mean=0.0,stddev=1.0,dtype=tf.float32),name=weightName,trainable=True)
    a = tf.Variable(tf.zeros([1]),name="a",trainable=True)
    #tf.histogram_summary(weightName,w1)
    tf.histogram_summary(weightName,a)
    net=tf.matmul(X,w1)#9by4
    z1, n2, n3, n4, n5, n6, n7, n8, n9 =tf.split(0,batch_size,net)#Must split like this
    z2=tf.add(n2,tf.mul(a,z1))#1by4
    z3=tf.add(n3,tf.mul(a,z2))
    z4=tf.add(n4,tf.mul(a,z3))
    z5=tf.add(n5,tf.mul(a,z4))
    z6=tf.add(n6,tf.mul(a,z5))
    z7=tf.add(n7,tf.mul(a,z6))
    z8=tf.add(n8,tf.mul(a,z7))
    z9=tf.add(n9,tf.mul(a,z8))
    z=tf.concat(0,[z1,z2,z3,z4,z5,z6,z7,z8,z9])
    return z
  
def GaussLayer(X,b2,batch_size,num_h2_weight,sigmaName,meanName,nameScope):
  with tf.name_scope(nameScope):
    sigma = tf.Variable(5*tf.ones([1]),name=sigmaName,trainable=True)
    mean = tf.Variable(tf.zeros([3,1]),name=meanName,trainable=True)
    b1, z1, z2 =tf.split(1,3,X)#Get the vectors that are 6x1
    m1, m2, m3 =tf.split(0,3,mean)
    e6=tf.ones([batch_size,1])
    x1=tf.sub(b1,tf.mul(m1,e6))
    x2=tf.sub(z1,tf.mul(m2,e6))
    x3=tf.sub(z2,tf.mul(m3,e6))
    den=tf.mul(tf.constant([-1],dtype=tf.float32),tf.square(sigma))
    temp1=tf.div(tf.square(x1),(den))  
    temp2=tf.div(tf.square(x2),(den))  
    temp3=tf.div(tf.square(x3),(den))  
    s1=tf.exp(tf.add_n([temp1,temp2,temp3]))   #May be exploding or vanishing here  
  return s1  
  
def SoftmaxLayer(X,num_weights):
  with tf.name_scope("SoftMax"):	    
    w1 = tf.Variable(tf.truncated_normal(shape=[num_weights,1],mean=0.0,stddev=1.0,dtype=tf.float32),name="SM_w1",trainable=True)
    w2 = tf.Variable(tf.truncated_normal(shape=[num_weights,1],mean=0.0,stddev=1.0,dtype=tf.float32),name="SM_w2",trainable=True)
    w3 = tf.Variable(tf.truncated_normal(shape=[num_weights,1],mean=0.0,stddev=1.0,dtype=tf.float32),name="SM_w3",trainable=True)
    w4 = tf.Variable(tf.truncated_normal(shape=[num_weights,1],mean=0.0,stddev=1.0,dtype=tf.float32),name="SM_w4",trainable=True)
    net1=tf.matmul(X,w1)
    y1=tf.exp(net1)
    net2=tf.matmul(X,w2)
    y2=tf.exp(net2)
    net3=tf.matmul(X,w3)
    y3=tf.exp(net3)
    net4=tf.matmul(X,w4)
    y4=tf.exp(net4)
    den=tf.add_n([y1,y2,y3,y4])
    z1=tf.div(y1,den)  
    z2=tf.div(y2,den) 
    z3=tf.div(y3,den)  
    z=tf.concat(1,[z1,z2,z3])
  return z

def Loss(output_placeholder, z_out):
  with tf.name_scope("Loss"):	
    temp1=tf.sub(output_placeholder,z_out)
    loss=tf.nn.l2_loss(temp1,name="loss")
  return loss 
  
  
	

def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Compute the gradients for a list of variables
  #grads_vars
  grads_vars=optimizer.compute_gradients(loss,tf.trainable_variables())
  
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  #train_op = optimizer.minimize(loss, global_step=global_step)
  trainableVars=tf.trainable_variables()
  
  train_op = optimizer.apply_gradients(grads_vars)
  
  tf.histogram_summary("gv",grads_vars[1][1])
  return train_op


def evaluation_test(z_out, desired_output_placeholder):
  """Evaluate the quality of the predictions at predicting the word.

  Args:
    logits: Prediction tensor, float - [batch_size, NUM_CLASSES].
    desired_output_placeholder: Labels tensor, float32 - [batch_size, NUM_CLASSES], with values in the
      range [0, 1].

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  #test=tf.constant([[1,2,3],[4,5,6],[7,8,9]])
  #Find the indices of the largest prediction probability in the prediction vector z_out
  #in the event of a tie, choose the element with the lowest indice
  predictions_vals, predictions_indices = tf.nn.top_k(z_out,1,name=None)
  
  #Find the indices of the largest desired prediction probability in the output_place_holder vector
  #in the event of a tie, choose the element with the lowest indice
  truth_vals, truth_indices = tf.nn.top_k(desired_output_placeholder,1,name=None)
  
  #Compute the Confusion matrix for the predictions and desired predictions
  confusion=makeConfusion(predictions_indices,truth_indices)
 
  
  return confusion
  
  
def Bhattacharyya_distances(sentence_size,z_out,output_placeholder):
	#with tf.name_scope("Bhattacharyya Distance"):
	temp=tf.sqrt(tf.sub(1.0,tf.reduce_sum(tf.sqrt(tf.mul(z_out,output_placeholder)),1)))
	Bhats=tf.reshape(temp,[sentence_size,1])
	return Bhats  
  
  
def makeConfusion(predictions_indices,truth_indices):
  with tf.name_scope("Confusion"):	 
    c00=getElement(predictions_indices,truth_indices,0,0)#predict0 true 0
    c11=getElement(predictions_indices,truth_indices,1,1)#predict1 true 1
    c22=getElement(predictions_indices,truth_indices,2,2)#predict2 true 2
    c01=getElement(predictions_indices,truth_indices,0,1)#predict0 true 1
    c02=getElement(predictions_indices,truth_indices,0,2)#predict0 true 2
    c10=getElement(predictions_indices,truth_indices,1,0)#...
    c12=getElement(predictions_indices,truth_indices,1,2)
    c20=getElement(predictions_indices,truth_indices,2,0)
    c21=getElement(predictions_indices,truth_indices,2,1)#predict2 true 1
    #truth along rows, prediction along columns
    c0=tf.pack([c00,c01,c02])
    c1=tf.pack([c10,c11,c12])
    c2=tf.pack([c20,c21,c22])
    confusion=tf.reshape(tf.concat(0,[c0,c1,c2]),[3,3],name="confusion")
    # Metrics
    N=tf.add_n([c00,c01,c02,c10,c11,c12,c20,c21,c22])
    # For a
    aTP = tf.add(c00,tf.constant([0]))
    aTN = tf.add(c11,c22)
    aFP = tf.add(c01,c02)
    aFN = tf.add(c10,c20)
    aSense = tf.div(aTP,tf.add(aTP,aFN))
    aPrecision = tf.div(aTP,tf.add(aTP,aFP))
    # For b
    bTP = tf.add(c11,tf.constant([0]))
    bTN = tf.add(c00,c22)
    bFP = tf.add(c10,c12)
    bFN = tf.add(c01,c21)
    bSense = tf.div(bTP,tf.add(bTP,bFN))
    bPrecision = tf.div(bTP,tf.add(bTP,bFP))
    # For c
    cTP = tf.add(c22,tf.constant([0]))
    cTN = tf.add(c00,c11)
    cFP = tf.add(c20,c21)
    cFN = tf.add(c02,c12)
    cSense = tf.div(cTP,tf.add(cTP,cFN))
    cPrecision = tf.div(cTP,tf.add(cTP,cFP))
   
  
    #Calculate Correct Classiication Ratio here CCR
    #CCR = tf.div((c00+c11+c22),N) #trace of confusion by N
    
    #Calculate Kappa here
    #Pr1=tf.div(c0,N)
    #Pr2=tf.div(c1,N)
    #Pr3=tf.div(c2,N)
    #Pc1=tf.div((c00+c10+c20),N)
    #Pc2=tf.div((c01+c11+c21),N)
    #Pc3=tf.div((c02+c12+c22),N)
    #PrPc=Pr1*Pc1+Pr2*Pc2+Pr3*Pc3
    #kappa=tf.div(tf.sub(CCR,PrPc),tf.sub(tf.constant([1]),PrPc))
  
    #Calculate FScore here 
    #Fa=tf.div(2*aPrecision*aSense,(aPrecision+aSense))
    #Fb=tf.div(2*bPrecision*bSense,(bPrecision+bSense))
    #Fc=tf.div(2*cPrecision*cSense,(cPrecision+cSense))
  
  return confusion
 
  
def getElement(predictions_indices,truth_indices,pred_word_index,truth_word_index):  
  with tf.name_scope("getElementConfusion"):
    #Number of predictions at the pred_word_index
    bool_pa=tf.equal(predictions_indices,tf.constant([pred_word_index]))
    #Number of truths at the pred_word_index
    bool_ta=tf.equal(truth_indices,tf.constant([truth_word_index]))
    #Concatenate the two vectors columnwise
    temp=tf.concat(1,[bool_pa,bool_ta])
    #Logical AND the matrix along the rows
    bool_pa_ta=tf.reduce_all(temp,reduction_indices=1,keep_dims=True,name=None)
    #Compute the number of matches between the prediction and truth
    num_pa_ta=tf.reduce_sum(tf.cast(bool_pa_ta, tf.int32)) 
  return num_pa_ta

def formatting(input_placeholder,desired_output_placeholder,z_out,Bhats,sentence_size):
		
	results_matrix=tf.concat(1,[z_out,Bhats])
	
	return results_matrix
	
def return_inputs(input_placeholder,desired_output_placeholder,z_out,Bhats,sentence_size):
		
	results_matrix=tf.concat(1,[input_placeholder,desired_output_placeholder])
	
	return results_matrix	




