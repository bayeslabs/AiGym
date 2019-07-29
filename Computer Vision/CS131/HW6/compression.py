import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    u, s, vh = np.linalg.svd(image)
    uu=u
    vvh=vh
    ite=np.shape(s)[0]-num_values
    for i in range(ite):
        m=0
        for j in range(np.shape(s)[0]):
           if(s[j]!=0 and s[j]<s[m]):
               m=j
        #u=np.delete(np.delete(u,m,0),m,1)
        #u[m]=0
        #u[:,m]=0
        #vh[m]=0
        #vh[:,m]=0
        #print(m)
        #u=np.delete(u,m,0)
        #s=np.delete(s,m)
        uu=np.delete(uu,m,0)
        vvh=np.delete(vvh,m,1)

        #vh=np.delete(vh,m,1)
        s[m]=0

    sm=np.zeros((np.shape(u)[1],np.shape(vh)[0]))
    for i in range(num_values):
        sm[i,i]=s[i]
    print("shape:",u.shape,sm.shape,vh.shape)
    compressed_image=np.dot(np.dot(u,sm),vh)
    compressed_size=np.dot(np.dot(uu,sm),vvh).size
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
