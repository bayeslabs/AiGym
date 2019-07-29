import numpy as np

def dot_product(vector1, vector2):
    """ Implement dot product of the two vectors.
    Args:
        vector1: numpy array of shape (x, n)
        vector2: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x,x) (scalar if x = 1)
    """
    out = None
    ### YOUR CODE HERE
    
    if(len(np.shape(vector1))==1):
        vector1.shape=(1,vector1.shape[0]);
    if(len(np.shape(vector2))==1):
        vector2.shape=(1,vector2.shape[0]);

    input_size1=np.shape(vector1);

    row=input_size1[0];
    
    out = np.ones((row,row));
    
    for i in range(row):
        for j in range(row):
            x1=vector1[i,];
            x2=vector2.T[j];
            out[i,j]=x1.dot(x2);

    if(np.shape(out)==(1,1)):
        out=out[0,0];
        
    ### END YOUR CODE
    return out

def matrix_mult(M, vector1, vector2):
    """ Implement (vector1.T * vector2) * (M * vector1)
    Args:
        M: numpy matrix of shape (x, n)
        vector1: numpy array of shape (1, n)
        vector2: numpy array of shape (n, 1)

    Returns:
        out: numpy matrix of shape (1, x)
    """
    out = None
    ### YOUR CODE HERE
    x=vector1.T*vector2;
    y=M*vector1;
    out=x.T.dot(y.T);
    ### END YOUR CODE

    return out

def svd(matrix):
    """ Implement Singular Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m)
        s: numpy array of shape (k)
        v: numpy array of shape (n, n)
    """
    u = None
    s = None
    v = None
    ### YOUR CODE HERE
    u,s,v=np.linalg.svd(matrix,1,1);
    ### END YOUR CODE

    return u, s, v

def get_singular_values(matrix, n):
    """ Return top n singular values of matrix
    Args:
        matrix: numpy matrix of shape (m, w)
        n: number of singular values to output
        
    Returns:
        singular_values: array of shape (n)
    """
    singular_values = None
    u, s, v = svd(matrix)
    ### YOUR CODE HERE
    singular_values=s[0:n];
    ### END YOUR CODE
    return singular_values

def eigen_decomp(matrix):
    """ Implement Eigen Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, )

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
    """
    w = None
    v = None
    ### YOUR CODE HERE
    w,v=np.linalg.eig(matrix);
    ### END YOUR CODE
    return w, v

def get_eigen_values_and_vectors(matrix, num_values):
    """ Return top n eigen values and corresponding vectors of matrix
    Args:
        matrix: numpy matrix of shape (m, m)
        num_values: number of eigen values and respective vectors to return
        
    Returns:
        eigen_values: array of shape (n)
        eigen_vectors: array of shape (m, n)
    """
    w, v = eigen_decomp(matrix)
    eigen_values = []
    eigen_vectors = []
    ### YOUR CODE HERE
    m=np.shape(matrix)[0];
    value_sorted=np.argsort(w);
    eigen_values=w[sorted(value_sorted[-num_values:])];
    eigen_vectors=np.zeros((m,num_values));
    for i in range(m):
        for j in range(num_values):
            eigen_vectors[i,j]=v[i,value_sorted[-1-j]];
    ### END YOUR CODE
    return eigen_values, eigen_vectors
