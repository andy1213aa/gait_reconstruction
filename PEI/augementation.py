import numpy as np
import cv2
num_channel = 5

def augmentation(X):
# [n,period,mx,my]
    Y = np.ones(X.shape)*X[0,0,0,0]
    Z = np.ones(X.shape)
    n = X.shape[0]
    for i in range(n):
        for t in range(num_channel):
            shift = np.random.randint(-8,8)
            if shift>0:
                Y[i,t,:,shift:64] = X[i,t,:,0:64-shift]
            else:
                Y[i,t,:,0:64+shift] = X[i,t,:,0-shift:64]     
            k1= np.random.randint(0,int(64*0.2))
            b = cv2.resize(Y[i,t,:,:],[64-k1,64-k1])
            k2 = np.random.randint(0,k1+1)
            Z[i,t,:,:] = Z[i,t,:,:]*b[0,0]
            Z[i,t,k2:k2+64-k1,k2:k2+64-k1] = b
    return Z/255.0