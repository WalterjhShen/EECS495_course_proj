#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from autograd import grad
import math
from autograd.misc.flatten import flatten_func
from autograd import numpy as np
import matplotlib.pyplot as plt

# In[ ]:


def batch_gradient_descent(g,x_sample,y_sample,alpha,max_its,w,batch_size,full_size):
    g_flat, unflatten_func, w_flat = flatten_func(g, w)
    gradient = grad(g_flat)

    # run the gradient descent loop
    weight_history = [w]
    epoch_history = [g_flat(w,x_sample,y_sample,np.arange(full_size))]           # container for epoch cost function history
    cost_history = [g_flat(w,x_sample,y_sample,np.arange(full_size))]          # container for corresponding cost function history
    for k in range(max_its):
        for j in range(math.ceil(float(full_size)/float(batch_size))):
            index = np.arange(j*batch_size,min((j+1)*batch_size,full_size))
            grad_eval = gradient(w,x_sample,y_sample,index)
            w = w - alpha*grad_eval
            cost_history.append(g_flat(w,x_sample,y_sample,np.arange(full_size)))
        epoch_history.append(g_flat(w,x_sample,y_sample,np.arange(full_size)))
        weight_history.append(w)
    return weight_history,epoch_history,cost_history

# compute C linear combinations of input point, one per classifier
def model(x,w):
    a = w[0] + np.dot(x.T,w[1:])
    return a.T

# multiclass perceptron
def multiclass_perceptron(w,x,y,iter):
    # get subset of points
    x_p = x[:,iter]
    y_p = y[:,iter]

    # pre-compute predictions on all points
    all_evals = model(x_p,w)

    # compute maximum across data points
    a =  np.max(all_evals,axis = 0)        

    # compute cost in compact form using numpy broadcasting
    b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
    cost = np.sum(a - b)

    # return average
    return cost/float(np.size(y_p))

def misclassification(x,y,w):
    all_evals = model(x,w)
    idx = np.argmax(all_evals,axis = 0)
    correct_numbers = np.sum(idx==y)/y.shape[1]
    return ((1.0-correct_numbers)*y.shape[1])

def get_mis(weight_history_1,x_sample,y_sample):
    mis_1=[]
    for i in range(len(weight_history_1)):
        mis_1.append(misclassification(x_sample,y_sample,weight_history_1[i]))
    return(mis_1,(y_sample.shape[1]-mis_1[-1])/y_sample.shape[1])

def plot_batch(epoch_history_1,epoch_history_2,mis_1,mis_2,pre_1,pre_2):
    fig = plt.figure(figsize=(15,5))

    ax1 = plt.subplot(131)
    ax1.set_xlabel('full epochs')
    ax1.set_ylabel('$g(w^k)$')
    ax1.plot(epoch_history_1,markersize=10,label='full batch')
    ax1.plot(epoch_history_2,markersize=10,label='minibatch-size=200')
    ax1.set_title('cost history_perceptron')
    ax1.legend()

    ax2 = plt.subplot(132)
    ax2.plot(mis_1,label='full batch')
    ax2.plot(mis_2,label='minibatch-size=200')
    ax2.set_xlabel('full epochs')
    ax2.set_ylabel('misclassification number')
    ax2.set_title('misclassification history')
    ax2.legend()

    ax3 = plt.subplot(133)
    ax3.plot(pre_1,label='full batch')
    ax3.plot(pre_2,label='minibatch-size=200')
    ax3.set_xlabel('full epochs')
    ax3.set_ylabel('misclassification number')
    ax3.set_title('prediction misclassification history')
    ax3.legend()

    return(plt.show())


def gradient_descent(g,alpha,max_its,w,batch_size,full_size):
    g_flat, unflatten_func, w_flat = flatten_func(g, w)
    gradient = grad(g_flat)

    # run the gradient descent loop
    weight_history = [w]
    cost_history = [g_flat(w,np.arange(full_size))]          # container for corresponding cost function history
    for k in range(max_its):
        for j in range(math.ceil(float(full_size)/float(batch_size))):
            index = np.arange(j*batch_size,min((j+1)*batch_size,full_size))
            grad_eval = gradient(w,index)
            w = w - alpha*grad_eval
            cost_history.append(g_flat(w,np.arange(full_size)))
        weight_history.append(w)
    return weight_history,cost_history

# multiclass softmaax cost
def multiclass_softmax(w,x,y,iter):
    x_p = x[:,iter]
    y_p = y[:,iter]
    # pre-compute predictions on all points
    all_evals = model(x_p,w)
    
    # compute softmax across data points
    a = np.log(np.sum(np.exp(all_evals),axis = 0)) 
    
    # compute cost in compact form using numpy broadcasting
    b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
    cost = np.sum(a - b)
    
    # return average
    return cost/float(np.size(y_p))

# standard normalization function 
def standard_normalizer(x):
    # compute the mean and standard deviation of the input
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_stds = np.std(x,axis = 1)[:,np.newaxis]   
    x_stds[x_stds==0] = 1

    # create standard normalizer function
    normalizer = lambda data: (data - x_means)/x_stds

    # create inverse standard normalizer
    inverse_normalizer = lambda data: data*x_stds + x_means

    # return normalizer 
    return normalizer

# compute eigendecomposition of data covariance matrix for PCA transformation
def PCA(x,**kwargs):
    # regularization parameter for numerical stability
    lam = 10**(-7)
    if 'lam' in kwargs:
        lam = kwargs['lam']

    # create the correlation matrix
    P = float(x.shape[1])
    Cov = 1/P*np.dot(x,x.T) + lam*np.eye(x.shape[0])

    # use numpy function to compute eigenvalues / vectors of correlation matrix
    d,V = np.linalg.eigh(Cov)
    return d,V

# PCA-sphereing - use PCA to normalize input features
def PCA_sphereing(x,**kwargs):
    # Step 1: mean-center the data
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_centered = x - x_means

    # Step 2: compute pca transform on mean-centered data
    d,V = PCA(x_centered,**kwargs)

    # Step 3: divide off standard deviation of each (transformed) input, 
    # which are equal to the returned eigenvalues in 'd'.  
    stds = (d[:,np.newaxis])**(0.5)
    normalizer = lambda data: np.dot(V.T,data - x_means)/stds

    # create inverse normalizer
    inverse_normalizer = lambda data: np.dot(V,data*stds) + x_means

    # return normalizer 
    return normalizer

def plot_result(cost_history_1,cost_history_2,cost_history_3,mis_1,mis_2,mis_3,pre_1,pre_2,pre_3):
    
    fig = plt.figure(figsize=(15,5))

    ax1 = plt.subplot(131)
    ax1.set_xlabel('step k')
    ax1.set_ylabel('$g(w^k)$')
    ax1.plot(cost_history_1,markersize=10,label='original')
    ax1.plot(cost_history_2,markersize=10,label='standard')
    ax1.plot(cost_history_3,markersize=10,label='sphered')
    ax1.set_title('cost history')
    ax1.legend()

    ax2 = plt.subplot(132)
    ax2.plot(mis_1,label='original')
    ax2.plot(mis_2,label='standard')
    ax2.plot(mis_3,label='sphered')
    ax2.set_xlabel('step k')
    ax2.set_ylabel('misclassification')
    ax2.set_title('misclassification history')
    ax2.legend()

    ax3 = plt.subplot(133)
    ax3.plot(pre_1,label='original')
    ax3.plot(pre_2,label='standard')
    ax3.plot(pre_3,label='sphered')
    ax3.set_xlabel('step k')
    ax3.set_ylabel('misclassification')
    ax3.set_title('misclassification history')
    ax3.legend()

    return(plt.show())
    