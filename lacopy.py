                   LINEAR ALGEBRA 

Practical – 1 

Aim:  

Write a program which demonstrates the following:  
a. Addition of two complex numbers 

Code: 

x = 1 + 3j 

y = 10 + 3j 

print("Addition of two complex numbers:", x+y) 

Output: 

Addition of two complex numbers: (11+6j) 

 

 b. Displaying the conjugate of a complex number  

Code: 

a = 4 + 2j 

print("Conjugate of a given complex number:",a.conjugate()) 

Output: 

Conjugate of a given complex number: (4-2j) 

 
c. Plotting a set of complex numbers  

Code: 

x = 2 + 2j 

a = [-2+4j,-1.5+3j,-1+5j,0+2j,1+1.5j] 

X = [x.real for x in a] 

Y = [x.imag for x in a] 

plt.scatter(x,y,color='blue') 

plt.show() 

 
d. Creating a new plot by rotating the given number by a degree 90, 180, 270 degrees and also by scaling by a number a = 1/2, a = 1/3, a = 2 etc. 

Code: 

import numpy as np 

import matplotlib.pyplot as plt 

 

# Define the original vector 

vector = np.array([3, 2])  # Replace with your desired vector 

 

# Rotation angles in degrees 

angles = [90, 180, 270]  # Replace with the angles you want to rotate by 

 

# Scaling factors 

scales = [1/2, 1/3, 2]  # Replace with the scaling factors you want 

 

# Create a figure and set axis limits 

plt.figure(figsize=(10, 5)) 

plt.xlim(-5, 5) 

plt.ylim(-5, 5) 

 

# Plot the original vector 

plt.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Original Vector') 

 

# Rotate and scale the vector 

for angle in angles: 

    rotated_vector = np.dot(np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))], 

                                      [np.sin(np.radians(angle)), np.cos(np.radians(angle))]]), vector) 

    plt.quiver(0, 0, rotated_vector[0], rotated_vector[1], angles='xy', scale_units='xy', scale=1, label=f'Rotated {angle} degrees') 

 

for scale in scales: 

    scaled_vector = vector * scale 

    plt.quiver(0, 0, scaled_vector[0], scaled_vector[1], angles='xy', scale_units='xy', scale=1, label=f'Scaled by {scale}') 

 

# Set axis equal for consistent scaling 

plt.gca().set_aspect('equal', adjustable='box') 

 

# Add legend and labels 

plt.legend() 

plt.xlabel('X') 

plt.ylabel('Y') 

plt.grid() 

plt.title('Vector Rotation and Scaling') 

 

# Show the plot 

plt.show() 

 

Extra activities: 

1)(1+3j)+(10+20j) 

Code: 

x = 1+3j 

y = 10+20j 

print("Addition of two complex numbers:",x+y) 

output: 

Addition of two complex numbers: (11+23j) 

 

2. If x=1+3j then find (x-1)**2 

Code: 

x = 1+3j 

y = (x-1)**2 

print("Value for y:",y) 

output: 

Value for y: (-9+0j) 

 

3. 1+2j*3 

Code: 

x = 1+2j*3 

print("Value for x:",x) 

output: 

Value for x: (1+6j) 

 

4. 4*3j**2 

Code: 

w=4*3j**2 

print("Value for w:",w) 

output: 

Value for w: (-36+0j) 

 

5. If x=1+3j the find x.real & x.imag 

Code: 

x = 1+3j 

a = [-2+4j,-1.5+3j,-1+5j,0+2j,1+1.5j] 

X = [x.real for x in a] 

Y = [y.real for x in a] 

plt.scatter(x,y,color='red') 

plt.show() 

 

6. If x=1+3j the find x.conjugate 

Code: 

x = 1+3j 

print("Conjugate of a given complex number:",x.conjugate()) 

Output: 

Conjugate of a given complex number: (1-3j) 

 

7. Plot S = {3+3i, 4+3i, 2+i, 2.5+i, 3+i, 3.25+i} 

Code: 

x = 1+3j 

s = [3+3j, 4+3j, 2+1j, 2.5+1j, 3+1j, 3.25+1j] 

X = [x.real for x in a] 

Y = [x.real for x in a] 

plt.scatter(x,y,color='red') 

plt.show() 

 

Practical – 2 

Aim:  

Write a program to do the following:  
1. Enter a vector u as a n-list  
2. Enter another vector v as a n-list   
3. Find the vector au + bv for different values of a and b  
4. Find the dot product of u and v 

Code: 

def addvec(x,y): 

    return[x[i]+y[i]for i in range(len(x))] 

def subvec(x,y): 

    return[x[i]-y[i]for i in range(len(x))] 

def scalarmul(x,p): 

    return[p*x[i]for i in range(len(x))] 

def dotprod(x,y): 

    return sum([x[i]*y[i] for i in range(len(x))]) 

v=[] 

u=[] 

n=int(input('Enter no. of elements you want to add in vector:')) 

print('Enter elements of vector u:') 

for i in range(n): 

    elem=int(input('Enter element:')) 

    u.append(elem) 

print('Vector u=',u) 

print('Enter elements of vector v:') 

for i in range(n): 

    elem=int(input('Enter element:')) 

    v.append(elem) 

print('Vector v=',v) 

while True: 

    print('Select vector operation') 

    print('1.Addition') 

    print('2.Subtraction') 

    print('3.Scalar Multiplication') 

    print('4.Dot Product') 

    print('5.Exit') 

    ch=int(input('Enter choice:')) 

    if ch==1: 

        print('Addition of Vectors u&v is(u+v)=',addvec(u,v)) 

    elif ch==2: 

        print('Substraction of vector u&v is(uv)=',subvec(u,v)) 

    elif ch==3: 

        print('To perform scalar multiplication au') 

        a=int(input('Enter value of a:')) 

        print('Scalar multiplication of au is',scalarmul(u,a)) 

    elif ch==4: 

        print('Dot product of u & v(u,v) is',dotprod(u,v)) 

    else: 

        break 

Output: 

Enter no. of elements you want to add in vector:2 

Enter elements of vector u: 

Enter element:1 

Enter element:2 

Vector u= [1, 2] 

Enter elements of vector v: 

Enter element:3 

Enter element:4 

Vector v= [3, 4] 

Select vector operation 

1.Addition 

2.Subtraction 

3.Scalar Multiplication 

4.Dot Product 

5.Exit 

Enter choice:1 

Addition of Vectors u&v is(u+v)= [4, 6] 

Select vector operation 

1.Addition 

2.Subtraction 

3.Scalar Multiplication 

4.Dot Product 

5.Exit 

Enter choice:2 

Substraction of vector u&v is(uv)= [-2, -2] 

Select vector operation 

1.Addition 

2.Subtraction 

3.Scalar Multiplication 

4.Dot Product 

5.Exit 

Enter choice:3 

To perform scalar multiplication au 

Enter value of a:5 

Scalar multiplication of au is [5, 10] 

Select vector operation 

1.Addition 

2.Subtraction 

3.Scalar Multiplication 

4.Dot Product 

5.Exit 

Enter choice:4 

Dot product of u & v(u,v) is 11 

Select vector operation 

1.Addition 

2.Subtraction 

3.Scalar Multiplication 

4.Dot Product 

5.Exit 

Enter choice:5 

 

Practical – 3 

Aim: 

Basic Matrix Operations:  
1. Matrix Addition, Subtraction, Multiplication  
2. Check if matrix is invertible.  
3. If yes then find Inverse 

Code: 

import numpy as np 

 

# Define two matrices A and B 

A = np.array([[1, 2], [3, 4]]) 

B = np.array([[5, 6], [7, 8]]) 

 

# Matrix Addition 

matrix_addition = A + B 

print("Matrix Addition:") 

print(matrix_addition) 

 

# Matrix Subtraction 

matrix_subtraction = A - B 

print("\nMatrix Subtraction:") 

print(matrix_subtraction) 

 

# Matrix Multiplication 

matrix_multiplication = np.dot(A, B) 

print("\nMatrix Multiplication:") 

print(matrix_multiplication) 

 

# Check if matrix A is invertible 

if np.linalg.det(A) != 0: 

    print("\nMatrix A is invertible.") 

     

    # Find the inverse of matrix A 

    A_inverse = np.linalg.inv(A) 

    print("\nInverse of Matrix A:") 

    print(A_inverse) 

else: 

    print("\nMatrix A is not invertible.") 

 

# Check if matrix B is invertible 

if np.linalg.det(B) != 0: 

    print("\nMatrix B is invertible.") 

     

    # Find the inverse of matrix B 

    B_inverse = np.linalg.inv(B) 

    print("\nInverse of Matrix B:") 

    print(B_inverse) 

else: 

    print("\nMatrix B is not invertible.") 

 

Output: 

Matrix Addition: 

[[ 6  8] 

 [10 12]] 

 

Matrix Subtraction: 

[[-4 -4] 

 [-4 -4]] 

 

Matrix Multiplication: 

[[19 22] 

 [43 50]] 

 

Matrix A is invertible. 

 

Inverse of Matrix A: 

[[-2.   1. ] 

 [ 1.5 -0.5]] 

 

Matrix B is invertible. 

 

Inverse of Matrix B: 

[[-4.   3. ] 

 [ 3.5 -2.5]] 

 

Practical – 4 

Aim: 

Basic Matrix Application – I  
Representation of Image in Matrix Format and Image Transformations 

Code: 

import cv2 

import numpy as np 

 

# Load an image 

image = cv2.imread('SEA.jpg') 

 

# Image Translation 

def translate_image(image, tx, ty): 

    rows, cols, _ = image.shape 

    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]]) 

    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows)) 

    return translated_image 

 

# Image Scaling 

def scale_image(image, scale_factor): 

    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR) 

    return scaled_image 

 

# Image Cropping 

def crop_image(image, x, y, width, height): 

    cropped_image = image[y:y+height, x:x+width] 

    return cropped_image 

 

# Example usage 

translated_image = translate_image(image, 50, 30) 

scaled_image = scale_image(image, 2.0) 

cropped_image = crop_image(image, 100, 100, 300, 200) 

 

# Save the results 

cv2.imwrite('translated_image.jpg', translated_image) 

cv2.imwrite('scaled_image.jpg', scaled_image) 

cv2.imwrite('cropped_image.jpg', cropped_image) 

 

# Display the images (optional) 

cv2.imshow('Translated Image', translated_image) 

cv2.imshow('Scaled Image', scaled_image) 

cv2.imshow('Cropped Image', cropped_image) 

cv2.waitKey(0) 

cv2.destroyAllWindows 

Output: Image 

 

Practical – 5 

Aim: 

Basic Matrix Application – II  
Perform Image addition, multiplication and subtraction 

Code: 

Addition: 

import cv2 

# Reading image files 

img1 = cv2.imread('sample-img-1.jpg') 

img2 = cv2.imread('sample-img-2.jpg') 

  

# Applying NumPy addition on images 

fimg = img1 + img2 

  

# Saving the output image 

cv2.imwrite('output.jpg', fimg) 

Subtraction: 

import cv2 

# Reading image files 

img1 = cv2.imread('sample-img-1.jpg') 

img2 = cv2.imread('sample-img-2.jpg') 

  

# Applying OpenCV subtraction on images 

fimg = cv2.subtract(img1, img2) 

  

# Saving the output image 

cv2.imwrite('output.jpg', fimg) 

Multiplication: 

import cv2 

# Reading image file 

img = cv2.imread('sample_img.jpg') 

  

# Applying NumPy scalar multiplication on image 

fimg = img * 1.5 

  

# Saving the output image 

cv2.imwrite('output.jpg', fimg) 

 

Practical – 6 

Aim: 

Create new plot by rotating the given number by 90, 180, 270 degrees. 
(i) Rotation by 90o 
To rotate a complex number by 90o, multiply z by i. 
 
(ii) Rotation by 180o 
To rotate a complex number by 180o, multiply z by -1. 
 
(c) Rotation by 270o  
To rotate a complex number by 270o, multiply z by -i. 

Code: 

import numpy as np 

z = 3+4j 

rotation_90 = z * 1j 

rotation_180 = z*(-1) 

rotation_270 = z*(-1j) 

print("Original complex number:",z) 

print("Rotated by 90 degrees:",rotation_90) 

print("Rotated by 180 degrees:",rotation_180) 

print("Rotated by 270 degrees:",rotation_270) 

output: 

Original complex number: (3+4j) 

Rotated by 90 degrees: (-4+3j) 

Rotated by 180 degrees: (-3-4j) 

Rotated by 270 degrees: (4-3j) 

 

Practical – 7 

Aim: 

Write a program to convert a matrix into its row echelon form. (Order 2). 

Code: 

def row_echelon_form(matrix): 

    # Check if the matrix is 2x2 

    if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2: 

        raise ValueError("Input matrix must be 2x2") 

 

    # Perform row operations to convert to row echelon form 

    if matrix[0][0] == 0: 

        matrix[0], matrix[1] = matrix[1], matrix[0]  # Swap rows if the first element is zero 

 

    if matrix[0][0] != 0: 

        factor = matrix[1][0] / matrix[0][0] 

        for i in range(2): 

            matrix[1][i] -= factor * matrix[0][i] 

 

    return matrix 

 

# Example 2x2 matrix 

matrix = [ 

    [2, 3], 

    [1, 4] 

] 

 

# Convert the matrix to its row echelon form 

ref_matrix = row_echelon_form(matrix) 

 

# Print the row echelon form 

for row in ref_matrix: 

    print(row) 

Output: 

[2, 3] 

[0.0, 2.5] 

 
Write a program to find rank of a matrix. 

Code: 

import numpy as np 

 

# Define the matrix 

matrix = [ 

    [1, 2, 3], 

    [4, 5, 6], 

    [7, 8, 9] 

] 

 

# Convert the matrix to a NumPy array 

matrix_array = np.array(matrix) 

 

# Find the rank of the matrix using numpy.linalg.matrix_rank 

rank = np.linalg.matrix_rank(matrix_array) 

 

# Print the rank 

print("Rank of the matrix:", rank) 

Output: 

Rank of the matrix: 2 

 

Practical – 8 

Aim: Write a program to calculate eigenvalue and eigenvector (Order 2 and 3) 

Code: 

import numpy as np 

 

a = np.mat("3 -2; 1 0") 

print("Matrix a:") 

print(a) 

 

eigenvalues = np.linalg.eigvals(a) 

print("Eigen values:", eigenvalues) 

 

eigenvalues, eigenvectors = np.linalg.eig(a) 

print("Eigenvalues:", eigenvalues) 

print("Eigenvectors:") 

print(eigenvectors) 

 

Output: 

Matrix a: 

[[ 3 -2] 

 [ 1  0]] 

Eigen values: [2. 1.] 

Eigenvalues: [2. 1.] 

Eigenvectors: 

[[0.89442719 0.70710678] 

 [0.4472136  0.70710678]] 

 

Practical – 9 

Aim: 

Implement Google’s Page rank algorithm. 

Code: 

import matplotlib.pyplot as plt 

import networkx as nx 

import pandas as pd 

import scipy as scipy 

Create Graph 

G = nx.DiGraph() 

 

[G.add_node(k) for k in ["A", "B", "C", "D", "E", "F", "G"]] 

G.add_edges_from([('G','A'), ('A','G'),('B','A'), 

                  ('C','A'),('A','C'),('A','D'), 

                  ('E','A'),('F','A'),('B','D'), 

                  ('D','F')]) 

pos = nx.spiral_layout(G) 

nx.draw(G, pos, with_labels = True, node_color="red") 

 

Run pagerank 

pr1 = nx.pagerank(G) 

print(pr1) 

nx.draw(G, pos, nodelist=list(pr1.keys()), node_size=[round(v * 4000) for v in pr1.values()],  

        with_labels = True, node_color="red")    

 

Adjusting dampening/teleports 

pr_09 = nx.pagerank(G, alpha=0.9) 

pr_08 = nx.pagerank(G, alpha=0.8) 

res = pd.DataFrame({"alpha=0.9": pr_09, "alpha=0.8": pr_08}) 

res 

 

Personal PageRank 

pr_e = nx.pagerank(G, alpha=0.9, personalization={'E': 1}) 

print(pr_e) 

nx.draw(G, pos, nodelist=list(pr_e.keys()), node_size=[round(v * 4000) for v in pr_e.values()],  

        with_labels = True, node_color="red")    

 

MovieLens Recommender  

Load data 

import urllib.request 

import shutil 

import zipfile 

import os 

filename='ml-100k' 

data_url='https://files.grouplens.org/datasets/movielens/ml-100k.zip' 

with urllib.request.urlopen(data_url) as response, open('./'+filename, 'wb') as out_file: 

    shutil.copyfileobj(response, out_file) 

print('Download completed') 

with zipfile.ZipFile('./'+filename, 'r') as zip_ref: 

    zip_ref.extractall('./sample_data/') 

dirs = [x[0] for x in os.walk("./sample_data")] 

ml = filter(lambda dirName: dirName if ('ml' in dirName) else '', list(dirs)) 

dt_dir_name= list(ml)[0] 

rdata = pd.read_csv(dt_dir_name +'/'+ 'u.data', delimiter='\t', names=['userId', 'movieId', 'rating', 'timestamp']) 

rdata['userId'] = 'u' + rdata['userId'].astype(str) 

rdata['movieId'] = 'i' + rdata['movieId'].astype(str) 

 

usrdata = pd.read_csv(dt_dir_name +'/'+'u.user', delimiter='|', names=['user id', 'age' ,'gender' ,'occupation' , 'zip code']) 

item_data = pd.read_csv(dt_dir_name +'/'+ 'u.item', delimiter='|', encoding="ISO-8859-1", header=None) 

item_data = item_data[[0,1]] 

item_data.columns = ['movieId','movieTitle'] 

item_data['movieId'] = 'i' + item_data['movieId'].astype(str) 

item_data = item_data.set_index('movieId') 

rdata.head() 

rdata = pd.merge(rdata, item_data, how='left', on='movieId') 

rdata.head() 

 

Create graph 

#Create a graph 

G = nx.Graph() 

#Add nodes 

G.add_nodes_from(rdata.userId, bipartite=0) 

G.add_nodes_from(rdata.movieId, bipartite=1) 

#Add weights for edges 

G.add_weighted_edges_from([(uId, mId,rating) for (uId, mId, rating) 

              in rdata[['userId', 'movieId', 'rating']].to_numpy()]) 

print(nx.info(G)) 

print(nx.is_bipartite(G)) 

 

Run pagerank 

movie_rank = nx.pagerank(G, alpha=0.85) 

def return_top_movies(movie_rank): 

  movie_rank = dict(sorted(movie_rank.items(), key=lambda item: item[1], reverse=True)) 

  top_10_movies = [] 

  for key, value in movie_rank.items(): 

    if 'i' in key: 

      top_10_movies.append(key) 

      if len(top_10_movies) == 10: 

        break 

  return item_data.loc[top_10_movies] 

return_top_movies(movie_rank) 

 

 

 

 

Practical – 10 

Aim: 

Write a program to do the following:  
a. Enter a vector b and find the projection of b orthogonal to a given vector u. 

Code: 

import numpy as np 

 

# Function to find the projection of vector b orthogonal to vector u 

def orthogonal_projection(b, u): 

    # Convert vectors to NumPy arrays for easier calculations 

    b = np.array(b) 

    u = np.array(u) 

     

    # Calculate the dot product of b and u 

    dot_product = np.dot(b, u) 

     

    # Calculate the dot product of u with itself 

    u_dot_u = np.dot(u, u) 

     

    # Calculate the projection 

    projection = b - (dot_product / u_dot_u) * u 

     

    return projection 

 

# Input vectors b and u 

b = [3, 1]  # Replace with your own vector b 

u = [4, 2]  # Replace with the vector u 

 

# Calculate the projection of b orthogonal to u 

projection = orthogonal_projection(b, u) 

 

# Print the result 

print("Vector b:", b) 

print("Vector u:", u) 

print("Projection of b orthogonal to u:", projection) 

Output: 

Vector b: [3, 1] 

Vector u: [4, 2] 

Projection of b orthogonal to u: [ 0.2 -0.4] 

  
b. Find the projection of b orthogonal to a set of given vectors 

code: 

import numpy as np 

 

# Function to perform Gram-Schmidt orthogonalization 

def gram_schmidt(vectors): 

    num_vectors, vector_dim = vectors.shape 

    ortho_basis = np.zeros((num_vectors, vector_dim), dtype=float) 

     

    for i in range(num_vectors): 

        ortho_vector = vectors[i] 

        for j in range(i): 

            projection = np.dot(vectors[i], ortho_basis[j]) / np.dot(ortho_basis[j], ortho_basis[j]) 

            ortho_vector -= projection * ortho_basis[j] 

        ortho_basis[i] = ortho_vector 

     

    return ortho_basis 

 

# Function to find the projection of vector b orthogonal to a set of vectors 

def orthogonal_projection(b, vectors): 

    b = np.array(b, dtype=float) 

    vectors = np.array(vectors, dtype=float) 

     

    # Perform Gram-Schmidt orthogonalization on the given vectors 

    ortho_basis = gram_schmidt(vectors) 

     

    # Calculate the projection of b onto the orthogonal basis 

    projection = np.zeros_like(b) 

     

    for v in ortho_basis: 

        # Check if the vector in the orthogonal basis is not a zero vector 

        if not np.allclose(v, np.zeros_like(v)): 

            projection += (np.dot(b, v) / np.dot(v, v)) * v 

     

    orthogonal_projection = b - projection 

     

    return orthogonal_projection 

 

# Define a set of vectors (as rows in a NumPy array) 

vectors = np.array([ 

    [1, 0], 

    [0, 1], 

    [1, 1] 

], dtype=float) 

 

# Define vector b 

b = np.array([2, 3], dtype=float) 

 

# Calculate the projection of b orthogonal to the set of vectors 

projection = orthogonal_projection(b, vectors) 

 

print("Original Vector b:", b) 

print("Set of Vectors:") 

print(vectors) 

print("Projection of b orthogonal to the set of vectors:") 

print(projection) 

output: 

Original Vector b: [2. 3.] 

Set of Vectors: 

[[1. 0.] 

 [0. 1.] 

 [1. 1.]] 

Projection of b orthogonal to the set of vectors: 

[0. 0.] 

 

Practical – 11 

Aim: 

Vector Applications: Classify given data using support vector machines (SVM) 

Code: 

Necessary imports  

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

 

Load Data from CSV File  

cell_df = pd.read_csv('D:\JYOTI\MVLU\LECTURES\SYCS\Test_Jupyter\cell_samples.csv ') 

# print first 5 line head() 

#cell_df.head()  

 

# print last 5 line tail() 

# cell_df.tail() 

 

# will display how many values are there in dataset 

# cell_df.shape 

 

# size in bytes 

# cell_df.size 

 

# display columnwise count of values available in each column 

# cell_df.count() 

 

# cell_df['Class'].value_counts() 

 

Distribution of the classes  

benign_df = cell_df[cell_df['Class']==2][0:200] 

malignant_df = cell_df[cell_df['Class']==4][0:200] 

 

#help(benign.df_plot) 

 

axes = benign_df.plot(kind='scatter', x='Clump', y='UnifSize', color='blue', label='Benign') 

malignant_df.plot(kind='scatter', x='Clump', y='UnifSize', color='red', label='Malignant', ax=axes) 

 

Identifying unwanted rows  

cell_df.dtypes 

 

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()] 

cell_df['BareNuc'] = cell_df['BareNuc'].astype('int') 

cell_df.dtypes 

 

Remove unwanted columns 

cell_df.columns 

 

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 

       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']] 

 

# cell_df 100 rows and 11 columns, 

# picked 9 columns out of 11 

 

# Independent var  

X = np.asarray(feature_df) 

 

# dependent variable 

y = np.asarray(cell_df['Class']) 

 

X[0:5] 

 

y[0:5] 

 

Divide the data as Train/Test dataset  

from sklearn.model_selection import train_test_split 

 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4) 

 

# 546 x 9 

X_train.shape 

# 546 x 1 

y_train.shape 

 

# 137 x 9 

X_test.shape 

# 137 x 1 

y_test.shape 

 

Modeling (SVM with Scikit-learn) from sklearn  

from sklearn import svm 

classifier = svm.SVC(kernel='linear', gamma='auto', C=2) 

classifier.fit(X_train, y_train) 

y_predict = classifier.predict(X_test) 

 

Evaluation (Results) 

from sklearn.metrics import classification_report 

print(classification_report(y_test, y_predict)) 

 

 
