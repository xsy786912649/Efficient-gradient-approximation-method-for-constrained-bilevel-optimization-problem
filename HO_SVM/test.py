import cvxpy as cp
import numpy as np
import time
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.nn import functional as F

def subsets(nums):
    ans = [[]]
    for i in nums:
        l = len(ans)
        for j in range(l):
            t = []
            t.extend(ans[j])
            t.append(i)
            ans.append(t)
    return ans

print(subsets([1,2,3]))