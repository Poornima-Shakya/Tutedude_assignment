# Objective - Develop a deeper understanding of matrices and vectors, including advanced operations, practical linear algebra concepts, and their application in data science.


# Instructions - Install the numpy library (if not already installed) by running: pip install numpy


# 1. Matrix and Vector Operations


# 1. Create a 3 × 3 matrix A and a 3 × 1 vector B :


#    import numpy as np
#    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#    B = np.array([1, 2, 3])


#    Tasks -
#    - Perform matrix-vector multiplication  A X B.
#    - Calculate the trace of matrix  A  (sum of diagonal elements).
#    - Find the eigenvalues and eigenvectors of A.


# 2. Replace the last row of matrix A with [10, 11, 12] and:
#    - Compute the determinant of the updated matrix A.
#    - Identify if the updated matrix is singular or non-singular.


import numpy as np

# Create a 3x3 matrix A and a 3x1 vector B
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[1], [2], [3]])

# 1.1 Perform matrix-vector multiplication A * B
product = np.dot(A, B)
print("Matrix-Vector Product A * B:\n", product)

# 1.2 Calculate the trace of matrix A
trace_A = np.trace(A)
print("Trace of A:", trace_A)

# 1.3 Find eigenvalues and eigenvectors of A
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues of A:", eigenvalues)
print("Eigenvectors of A:\n", eigenvectors)

# Update the last row of matrix A
A[-1] = [10, 11, 12]
print("Updated Matrix A:\n", A)

# 1.4 Compute the determinant of the updated matrix A
det_A = np.linalg.det(A)
print("Determinant of updated A:", det_A)

# 1.5 Check if the updated matrix is singular
is_singular = np.isclose(det_A, 0)
print("Is the updated matrix singular?", is_singular)

# 2. Invertibility of Matrices


# 1. Verify the invertibility of the updated matrix A:
#    - Check if the determinant is non-zero.
#    - If invertible, calculate the inverse of A.


# 2. Solve a system of linear equations A x X = B, where:
#    - A is the updated matrix.
#    -  	 



# Verify the invertibility of the updated matrix A
if not is_singular:
    # Calculate the inverse of A
    A_inv = np.linalg.inv(A)
    print("\nInverse of A:\n", A_inv)
else:
    print("Matrix A is singular, cannot compute inverse.")

# Solve the system of linear equations A x X = B
try:
    X = np.linalg.solve(A, B)
    print("Solution X to A X = B:\n", X)
except np.linalg.LinAlgError as e:
    print("Error solving linear equations:", e)



# 3. Practical Matrix Operations


# 1. Create a 4 × 4 matrix C with random integers between 1 and 20:
  
#    C = np.random.randint(1, 21, size=(4, 4))


#    Tasks
#    - Compute the rank of C.
#    - Extract the submatrix consisting of the first 2 rows and last 2 columns of C.
#    - Calculate the Frobenius norm of C.


# 2. Perform matrix multiplication between A (updated to 3 × 3) and C (trimmed to 3 × 3):
#    - Check if the multiplication is valid. If not, reshape C to make it compatible with A.


#  Create a 4x4 matrix C with random integers between 1 and 20
C = np.random.randint(1, 21, size=(4, 4))
print("Matrix C:\n", C)

# Compute the rank of C
rank_C = np.linalg.matrix_rank(C)
print("Rank of C:", rank_C)

# Extract the submatrix consisting of the first 2 rows and last 2 columns of C
submatrix_C = C[:2, -2:]
print("Submatrix of C:\n", submatrix_C)

# Calculate the Frobenius norm of C
frobenius_norm_C = np.linalg.norm(C, 'fro')
print("Frobenius Norm of C:", frobenius_norm_C)

# Perform matrix multiplication between the updated A and C
# Ensure C is 3x3 for multiplication
if C.shape[0] != 3 or C.shape[1] != 3:
    C = C[:3, :3]  # Trimming to 3x3 if necessary

# Check if the multiplication is valid and perform it
if A.shape[1] == C.shape[0]:
    multiplication_result = np.dot(A, C)
    print("Result of A * C:\n", multiplication_result)
else:
    print("Matrix multiplication is not valid, check dimensions of A and C.")

# 4. Data Science Context


# 1. Create a dataset as a 5 × 5 matrix D, where each column represents a feature, and each row represents a data point:


#    D = np.array([[3, 5, 7, 9, 11],
#                  [2, 4, 6, 8, 10],
#                  [1, 3, 5, 7, 9],
#                  [4, 6, 8, 10, 12],
#                  [5, 7, 9, 11, 13]])


#    Tasks
#    - Standardize D column-wise (mean = 0, variance = 1).
#    - Compute the covariance matrix of D.
#    - Perform Principal Component Analysis (PCA):
#      - Find the eigenvalues and eigenvectors of the covariance matrix.
#      - Reduce D to 2 principal components.

#Create a dataset as a 5x5 matrix D
D = np.array([[3, 5, 7, 9, 11],
              [2, 4, 6, 8, 10],
              [1, 3, 5, 7, 9],
              [4, 6, 8, 10, 12],
              [5, 7, 9, 11, 13]])

# Standardize D column-wise
D_mean = np.mean(D, axis=0)
D_std = np.std(D, axis=0)
D_standardized = (D - D_mean) / D_std
print("Standardized D:\n", D_standardized)

#  Compute the covariance matrix of D
cov_matrix_D = np.cov(D_standardized, rowvar=False)
print("Covariance Matrix of D:\n", cov_matrix_D)

#  Perform Principal Component Analysis (PCA)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix_D)
print("Eigenvalues of the Covariance Matrix:\n", eigenvalues)
print("Eigenvectors of the Covariance Matrix:\n", eigenvectors)

#  Reduce D to 2 principal components
# Projecting the standardized data onto the first two principal components
principal_components = eigenvectors[:, :2]  # Taking first 2 eigenvector columns
D_reduced = D_standardized @ principal_components
print("Reduced D to 2 Principal Components:\n", D_reduced)
