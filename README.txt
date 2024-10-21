Jayam Sutariya
CS 422

Project 4 (Write-Up)

PCA (pca.py)
Functions implemented:
def compute_Z(X, centering=True, scaling=False)
Computes the standardized Z based on the variables centering and scaling. If centering is true, the mean is 
subtracted from the data. If scaling is true, the data is divided by the standard deviation.

def compute_covariance_matrix(Z)
Computes the covariance matrix by dotting the transpose of the standardized Z with the standardized Z.

def find_pcs(COV)
Computes the Principal components (eigenvalues and eigenvectors) based on the covariance matrix using the 
built-in numpy function.

def project_data(Z, PCS, L, k, var)
Projects the standardized data onto a lower dimension using the eigenvalues and eigenvectors. The number of 
principal components maintained are decided by either k or var. If k=0, var, the desired cumulative variance, 
is used. Otherwise, k principal components are used. Projects the data by dotting the standardized data with
the transpose of the k principal components.


Application (compress.py)
Functions implemented:
def compress_images(DATA,k)
Takes in the data and k (number of principal components to maintain) and performs PCA on the data. The 
projected data is then compressed and rescaled to the correct dimensions. The rescaled images are then 
outputted to a output directory.

def load_data(input_dir)
Takes in the input directory and returns the content in a data matrix. Each file (image) in the directory 
is flattened and added to a matrix. The final matrix consists of all the images flattened together.