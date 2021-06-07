########################################################################
#                                                                      #
#                Matrix (2D) Linear Discriminant Analysis              #
#                        Author: Jeremy Walker                         #                        
#   Author Affiliation: San Jose State University College of Science   #
#                     Date Last Modified: 6/21/2021                    #     
#                     E-mail: jeremy.walker@sjsu.edu                   #       
#                                                                      #     
########################################################################

import numpy as np

#X_train should be the numeric training data in a numpy matrix, with each row corresponding 
#to a data point
#X_test should be in the same form as X_train
#y_train should be a 1-dimensional numpy array of integers to code for each class
#b is the dimension of the square matrix to reduce X_train and X_test down to; i.e. if b = 6
#then the output matrices will be 6x6

#NOTE: this algorithm does require the original matrix data from which X_train and X_test are
#created from be square, and can only return square output in the shape (b,b)

def matrixLDA(X_train, X_test, y_train, b):
    n_train = X_train.shape[0]; n_test = X_test.shape[0]
    n_pixels = int(np.sqrt(X_train.shape[1]))
    n_classes = len(set(y_train))
    
    X_train_tensor = np.zeros((n_pixels,n_pixels,n_train))
    for i in range(n_train):
        X_train_tensor[:,:,i] = np.reshape(X_train[i,:],(n_pixels,n_pixels))
        
    X_test_tensor = np.zeros((n_pixels,n_pixels,n_test))
    for i in range(n_test):
        X_test_tensor[:,:,i] = np.reshape(X_test[i,:],(n_pixels,n_pixels))
    
    Mj = np.zeros((n_pixels,n_pixels,n_classes))
    for j in range(n_classes):
        classSubset = X_train_tensor[:,:,y_train == j]
        Mj[:,:,j] = np.mean(classSubset, axis = 2)
    
    M = np.mean(Mj, axis = 2)
    
    R = np.identity(b)
    R = np.concatenate((R,np.zeros((n_pixels-b,b))), axis = 0)
    
    while(True):    
        SwR = np.zeros((n_pixels,n_pixels))
        for j in range(n_classes):
            classSubset = X_train_tensor[:,:,y_train == j]
            for i in range(classSubset.shape[2]):
                SwR = SwR + np.dot(np.dot(np.dot((classSubset[:,:,i]-Mj[:,:,j]),R),R.T),(classSubset[:,:,i]-Mj[:,:,j]).T)
        
        SbR = np.zeros((n_pixels,n_pixels))
        for j in range(n_classes):
            SbR = SbR + np.dot(np.dot(np.dot((Mj[:,:,j]-M),R),R.T),(Mj[:,:,j]-M).T) * sum(y_train == j)
        
        eigVals, eigVecs = np.linalg.eig(np.dot(np.linalg.inv(SwR),SbR))
        L = eigVecs[:,:b]
        
        SwL = np.zeros((n_pixels,n_pixels))
        for j in range(n_classes):
            classSubset = X_train_tensor[:,:,y_train == j]
            for i in range(classSubset.shape[2]):
                SwL = SwL + np.dot(np.dot(np.dot((classSubset[:,:,i]-Mj[:,:,j]).T,L),L.T),(classSubset[:,:,i]-Mj[:,:,j]))
    
        SbL = np.zeros((n_pixels,n_pixels))
        for j in range(n_classes):
            SbL = SbL + np.dot(np.dot(np.dot((Mj[:,:,j]-M).T,L),L.T),(Mj[:,:,j]-M)) * sum(y_train == j)
        
        eigVals, eigVecs = np.linalg.eig(np.dot(np.linalg.inv(SwL),SbL))
        newR = eigVecs[:,:b]
        
        Tolerance = np.sum(abs(newR-R)<0.001)
        if(Tolerance < n_pixels*b):
            R = newR
        else:
            break
    
    X_train_proj = np.zeros((n_train,b**2))
    for i in range(n_train):
        X_train_proj[i,:] = np.reshape(np.dot(np.dot(L.T,X_train_tensor[:,:,i]),R),(-1,b**2))
        
    X_test_proj = np.zeros((n_test,b**2))
    for i in range(n_test):
        X_test_proj[i,:] = np.reshape(np.dot(np.dot(L.T,X_test_tensor[:,:,i]),R),(-1,b**2))
        
    return X_train_proj, X_test_proj
