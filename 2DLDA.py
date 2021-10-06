########################################################################
#                                                                      #
#                Matrix (2D) Linear Discriminant Analysis              #
#                        Author: Jeremy Walker                         #                        
#   Author Affiliation: San Jose State University College of Science   #
#                    Date Last Modified: 10/06/2021                    #     
#                     E-mail: jeremy.walker@sjsu.edu                   #       
#                                                                      #     
########################################################################

#X_train should be the numeric training data in a numpy matrix, with each row corresponding 
#to a data point
#X_test should be in the same form as X_train
#y_train should be a 1-dimensional numpy array of integers to code for each class
#b is the dimension of the square matrix to reduce X_train and X_test down to; i.e. if b = 6
#then the output matrices will be 6x6

#NOTE: this algorithm does require the original matrix data from which X_train and X_test are
#created from be square, and can only return square output in the shape (b,b)

import numpy as np

class matrixLDA:
    def __init__(self, b):
        self.b = b
        
    def fit(self, X_train, y_train):
        self.n_train = X_train.shape[0]
        self.n_pixels = int(np.sqrt(X_train.shape[1]))
        self.n_classes = len(set(y_train))
        
        self.X_train_tensor = np.reshape(X_train, (self.n_train,self.n_pixels,self.n_pixels))
        
        Mj = np.zeros((self.n_classes,self.n_pixels,self.n_pixels))
        for j in range(self.n_classes):
            classSubset = self.X_train_tensor[y_train==j,:,:]
            Mj[j,:,:] = np.mean(classSubset, axis = 0)
        
        M = np.mean(Mj, axis = 0)
        
        R = np.identity(self.b)
        R = np.concatenate((R,np.zeros((self.n_pixels-self.b,self.b))), axis = 0)
        
        while(True):    
            SwR = np.zeros((self.n_pixels,self.n_pixels))
            for j in range(self.n_classes):
                classSubset = self.X_train_tensor[y_train==j,:,:]
                for i in range(classSubset.shape[0]):
                    SwR = SwR + np.dot(np.dot(np.dot((classSubset[i,:,:]-Mj[j,:,:]),R),R.T),(classSubset[i,:,:]-Mj[j,:,:]).T)
            
            SbR = np.zeros((self.n_pixels,self.n_pixels))
            for j in range(self.n_classes):
                SbR = SbR + np.dot(np.dot(np.dot((Mj[j,:,:]-M),R),R.T),(Mj[j,:,:]-M).T) * sum(y_train == j)
            
            eigVals, eigVecs = np.linalg.eig(np.dot(np.linalg.inv(SwR),SbR))
            L = eigVecs[:,:self.b]
            
            SwL = np.zeros((self.n_pixels,self.n_pixels))
            for j in range(self.n_classes):
                classSubset = self.X_train_tensor[y_train==j,:,:]
                for i in range(classSubset.shape[0]):
                    SwL = SwL + np.dot(np.dot(np.dot((classSubset[i,:,:]-Mj[j,:,:]).T,L),L.T),(classSubset[i,:,:]-Mj[j,:,:]))
        
            SbL = np.zeros((self.n_pixels,self.n_pixels))
            for j in range(self.n_classes):
                SbL = SbL + np.dot(np.dot(np.dot((Mj[j,:,:]-M).T,L),L.T),(Mj[j,:,:]-M)) * sum(y_train == j)
            
            eigVals, eigVecs = np.linalg.eig(np.dot(np.linalg.inv(SwL),SbL))
            newR = eigVecs[:,:self.b]
            
            Tolerance = np.sum(abs(newR-R)<0.001)
            if(Tolerance < self.n_pixels*self.b):
                R = newR
            else:
                break
        self.R = newR
        self.L = L
        
    def project(self, X_test):
        self.n_test = X_test.shape[0]
        self.X_test_tensor = np.reshape(X_test, (self.n_test,self.n_pixels,self.n_pixels))
        n_test = X_test.shape[0]

        X_train_proj = np.zeros((self.n_train,self.b**2))
        for i in range(self.n_train):
            X_train_proj[i,:] = np.reshape(np.dot(np.dot(self.L.T,self.X_train_tensor[i,:,:]),self.R),(-1,self.b**2))
            
        X_test_proj = np.zeros((self.n_test,self.b**2))
        for i in range(n_test):
            X_test_proj[i,:] = np.reshape(np.dot(np.dot(self.L.T,self.X_test_tensor[i,:,:]),self.R),(-1,self.b**2))
            
        return X_train_proj, X_test_proj
            
