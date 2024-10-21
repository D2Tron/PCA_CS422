import numpy as np

def compute_Z(X, centering=True, scaling=False):
    #Calculate the mean in the column direction
    mean = X.mean(axis=0)
    #Calculate the standard deviation in the row direction
    std = X.std(axis=0)

    newX = X

    #If centering is True, subtract mean from the data
    if (centering==True):
        newX = newX - mean

    #If scaling is True, divide by the standard deviation
    if (scaling==True):
        newX = newX/std

    #Return standardized Z
    return newX

def compute_covariance_matrix(Z):
    #Comput the covariance matrix by dotting Z-transpose by Z
    covM = np.dot(Z.T, Z)
    return covM

def find_pcs(COV):
    #Use the built-in function to calculate the eigenvalues and eigenvectors using the covariance matrix
    values, vectors = np.linalg.eig(COV)
    return values, vectors

def project_data(Z, PCS, L, k, var):
    #If k=0, use the cumulative variance
    if (k==0):
        #Calculate the total eigenvalues
        totalE = np.sum(L)
        #Designate a current variance variable
        curr = 0
        #Count for eigenvalues
        count = 0
        #While the current variance is less than the cumulative variance...
        while (curr < var):
            #Reset current variance to 0
            curr = 0
            #Traverse a for loop to add eigenvales to the current variance based on the eigenvalue count
            for i in range(count):
                curr += L[i]
            #Calculate the variance by dividing it by t
            curr = curr/totalE
            #Increase the count
            count += 1
        #Once the condition is met, subtract count by 1 to get the k value
        k = count-1
    
    #Select the first k eigenvectors 
    u = PCS[:k]
    #Calculate Z star by dotting Z by u-transpose
    Zstar = np.dot(Z, u.T)

    return Zstar