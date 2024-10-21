import numpy as np
import os
import matplotlib.pyplot as plt
import pca

def compress_images(DATA,k):
    #Output folder name
    outputDir = "Output"
    #Check if output folder exists, if not, make one
    dirName = os.path.isdir(outputDir)
    if not dirName:
        os.mkdir(outputDir)
    
    #Perform PCA on the data
    Z = pca.compute_Z(DATA)
    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    #Perform extra dot products needed for compression according to the project document
    U_T = PCS.T
    U_T = U_T[:k]
    X_compressed = np.dot(Z_star, U_T).T

    #Traverse through all the data files...
    for i in range(len(X_compressed)):
        #Designate a name for the output data file
        outFile = outputDir + "/" + str(i+1) + ".png"
        #Reshape the new image to 255
        newImg = np.array([[255 * (X_compressed[i][h * 48 + w] - X_compressed[i].min()) / (X_compressed[i].max() - X_compressed[i].min()) for w in range(48)] for h in range(60)])
        #Save the new data file to the designated directory as a grayscale
        plt.imsave(outFile, newImg, cmap='gray')

    return 0

def load_data(input_dir):
    #Create an empty list
    data = []
    #Traverse the input directory items
    for file in os.listdir(input_dir):
        #Read the file, flatten it, convert it to a floating point, and append it to the empty list
        data.append(plt.imread(input_dir + file).flatten().astype(float))
    #Create a numpy array out of the list
    out = np.array(data)
    #Get the transpose of the array and return it
    out = out.T
    return out