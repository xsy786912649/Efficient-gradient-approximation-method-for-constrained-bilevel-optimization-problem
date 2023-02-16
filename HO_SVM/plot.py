import numpy as np
from math import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, random, time
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import breast_cancer_svm_corrept
import breast_cancer_svm_corrept_my
import breast_cancer_svm

val_loss_array=[]
test_loss_array=[]
val_acc_array=[]
test_acc_array=[]
for seed in range(20,60):
    val_loss_list,test_loss_list,val_acc_list,test_acc_list,time_computation=breast_cancer_svm_corrept.run(seed)
    val_loss_array.append(np.array(val_loss_list))
    test_loss_array.append(np.array(test_loss_list))
    val_acc_array.append(np.array(val_acc_list))
    test_acc_array.append(np.array(test_acc_list))
    time_computation=np.array(time_computation)
val_loss_array=np.array(val_loss_array)
test_loss_array=np.array(test_loss_array)
val_acc_array=np.array(val_acc_array)
test_acc_array=np.array(test_acc_array)

val_loss_mean=np.sum(val_loss_array,axis=0)/val_loss_array.shape[0]
val_loss_sd=np.sqrt(np.var(val_loss_array,axis=0))/2.0
test_loss_mean=np.sum(test_loss_array,axis=0)/test_loss_array.shape[0]
test_loss_sd=np.sqrt(np.var(test_loss_array,axis=0))/2.0

val_acc_mean=np.sum(val_acc_array,axis=0)/val_acc_array.shape[0]
val_acc_sd=np.sqrt(np.var(val_acc_array,axis=0))/2.0
test_acc_mean=np.sum(test_acc_array,axis=0)/test_acc_array.shape[0]
test_acc_sd=np.sqrt(np.var(test_acc_array,axis=0))/2.0


val_loss_array_1=[]
test_loss_array_1=[]
val_acc_array_1=[]
test_acc_array_1=[]
for seed in range(20,60):
    val_loss_list_1,test_loss_list_1,val_acc_list_1,test_acc_list_1,time_computation_1=breast_cancer_svm_corrept_my.run(seed)
    val_loss_array_1.append(np.array(val_loss_list_1))
    test_loss_array_1.append(np.array(test_loss_list_1))
    val_acc_array_1.append(np.array(val_acc_list_1))
    test_acc_array_1.append(np.array(test_acc_list_1))
    time_computation_1=np.array(time_computation_1)
val_loss_array_1=np.array(val_loss_array_1)
test_loss_array_1=np.array(test_loss_array_1)
val_acc_array_1=np.array(val_acc_array_1)
test_acc_array_1=np.array(test_acc_array_1)

val_loss_mean_1=np.sum(val_loss_array_1,axis=0)/val_loss_array_1.shape[0]
val_loss_sd_1=np.sqrt(np.var(val_loss_array_1,axis=0))/2.0
test_loss_mean_1=np.sum(test_loss_array_1,axis=0)/test_loss_array_1.shape[0]
test_loss_sd_1=np.sqrt(np.var(test_loss_array_1,axis=0))/2.0

val_acc_mean_1=np.sum(val_acc_array_1,axis=0)/val_acc_array_1.shape[0]
val_acc_sd_1=np.sqrt(np.var(val_acc_array_1,axis=0))/2.0
test_acc_mean_1=np.sum(test_acc_array_1,axis=0)/test_acc_array_1.shape[0]
test_acc_sd_1=np.sqrt(np.var(test_acc_array_1,axis=0))/2.0

val_loss_array_0=[]
test_loss_array_0=[]
val_acc_array_0=[]
test_acc_array_0=[]
for seed in range(1,20):
    val_loss_list_0,test_loss_list_0,val_acc_list_0,test_acc_list_0,time_computation_0=breast_cancer_svm.run(seed)
    val_loss_array_0.append(np.array(val_loss_list_0))
    test_loss_array_0.append(np.array(test_loss_list_0))
    val_acc_array_0.append(np.array(val_acc_list_0))
    test_acc_array_0.append(np.array(test_acc_list_0))
    time_computation_0=np.array(time_computation_0)
val_loss_array_0=np.array(val_loss_array_0)
test_loss_array_0=np.array(test_loss_array_0)
val_acc_array_0=np.array(val_acc_array_0)
test_acc_array_0=np.array(test_acc_array_0)

val_loss_mean_0=np.sum(val_loss_array_0,axis=0)/val_loss_array_0.shape[0]
val_loss_sd_0=np.sqrt(np.var(val_loss_array_0,axis=0))/2.0
test_loss_mean_0=np.sum(test_loss_array_0,axis=0)/test_loss_array_0.shape[0]
test_loss_sd_0=np.sqrt(np.var(test_loss_array_0,axis=0))/2.0

val_acc_mean_0=np.sum(val_acc_array_0,axis=0)/val_acc_array_0.shape[0]
val_acc_sd_0=np.sqrt(np.var(val_acc_array_0,axis=0))/2.0
test_acc_mean_0=np.sum(test_acc_array_0,axis=0)/test_acc_array_0.shape[0]
test_acc_sd_0=np.sqrt(np.var(test_acc_array_0,axis=0))/2.0

plt.rcParams.update({'font.size': 18})
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus']=False 
axis=time_computation
plt.figure(figsize=(8,6))
#plt.grid(linestyle = "--") 
ax = plt.gca()
plt.plot(axis,val_loss_mean,'-',marker=".",label="Training loss (GD)")
ax.fill_between(axis,val_loss_mean-val_loss_sd,val_loss_mean+val_loss_sd,alpha=0.2)
plt.plot(axis,test_loss_mean,'--',marker=".",label="Test loss (GD)")
ax.fill_between(axis,test_loss_mean-test_loss_sd,test_loss_mean+test_loss_sd,alpha=0.2)
plt.plot(axis,val_loss_mean_1,'-',marker="1",label="Training loss (GAM)")
ax.fill_between(axis,val_loss_mean_1-val_loss_sd_1,val_loss_mean_1+val_loss_sd_1,alpha=0.2)
plt.plot(axis,test_loss_mean_1,'--',marker="1",label="Test loss (GAM)")
ax.fill_between(axis,test_loss_mean_1-test_loss_sd_1,test_loss_mean_1+test_loss_sd_1,alpha=0.2)
#plt.xticks(np.arange(0,iterations,40))
#plt.title('(c) Cumulative Rewards')
plt.xlabel('Running time /s')
#plt.legend(loc=4)
plt.ylabel("Loss")
#plt.xlim(-0.5,3.5)
#plt.ylim(0.5,1.0)
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('breast_1_clean_compare.pdf') 
plt.show()

axis=time_computation
plt.figure(figsize=(8,6))
ax = plt.gca()
plt.plot(axis,val_acc_mean,'-',marker=".",label="Training accuracy (GD, p=0.4)")
ax.fill_between(axis,val_acc_mean-val_acc_sd,val_acc_mean+val_acc_sd,alpha=0.2)
plt.plot(axis,test_acc_mean,'--',marker=".",label="Test accuracy (GD, p=0.4)")
ax.fill_between(axis,test_acc_mean-test_acc_sd,test_acc_mean+test_acc_sd,alpha=0.2) 
plt.plot(axis,val_acc_mean_1,'-',marker="1",label="Training accuracy (GAM, p=0.4)")
ax.fill_between(axis,val_acc_mean_1-val_acc_sd_1,val_acc_mean_1+val_acc_sd_1,alpha=0.2)
plt.plot(axis,test_acc_mean_1,'--',marker="1",label="Test accuracy (GAM, p=0.4)")
ax.fill_between(axis,test_acc_mean_1-test_acc_sd_1,test_acc_mean_1+test_acc_sd_1,alpha=0.2) 
plt.plot(axis,val_acc_mean_0,':' ,label="Training accuracy (p=0)")
ax.fill_between(axis,val_acc_mean_0-val_acc_sd_0,val_acc_mean_0+val_acc_sd_0,alpha=0.2)
plt.plot(axis,test_acc_mean_0,'-.',label="Test accuracy (p=0)")
ax.fill_between(axis,test_acc_mean_0-test_acc_sd_0,test_acc_mean_0+test_acc_sd_0,alpha=0.2) 
#plt.xticks(np.arange(0,iterations,40))
#plt.title('(c) Cumulative Rewards')
plt.xlabel('Running time /s')
plt.ylabel("Accuracy")
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('breast_2_clean_compare.pdf') 
plt.show()