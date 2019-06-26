#importing used libraries

from PIL import Image, ImageFilter
import numpy as np
import numpy.linalg as la
from numpy import dot
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import matplotlib.cm as cm
import scipy.ndimage.filters as fi
from scipy import sparse
import scipy.optimize as optimize
from scipy.ndimage import imread
import scipy
from pytictoc import TicToc

# make an image become grey

def Turn_grey(image):

    image_grey = .21*image[:,:,0] + .72*image[:,:,1] + .07*image[:,:,2]
    
    return image_grey

# create a gaussian kernel

def gaussian_kernel(k_len = 5, sigma = 3):
    d_mat = np.zeros((k_len, k_len))
    d_mat[k_len//2, k_len//2] = 1
    return fi.gaussian_filter(d_mat, sigma)

# get a blurred image

def getBlurredImage(image, n, sigma):

    kernel = gaussian_kernel(n, sigma)

    blurred_image = convolve2d(image, kernel)

    return blurred_image

# print an image

def printImage(image, N):

    plt.imshow(image.reshape((N,)*2), cmap="Greys_r")
    plt.show()

# find a toeplitz matrix

def toeplitz(b, n):
    m = len(b)
    T = np.zeros((n+m-1, n))
    for i in range(n+m-1):
        for j in range(n):
            if 0 <= i-j < m:
                T[i,j] = b[i-j]
    return T

# find col_mat(definned in article)

def col_mat(m, N, T):

    I = sparse.eye(N+m-1)

    return sparse.kron(T, I)

# find row_mat(definned in article)

def row_mat(N, T):

    I = sparse.eye(N)

    return sparse.kron(I, T)

# create a one dimensional gaussian

def gaussian1d(k_len = 5, sigma = 3):
    return gaussian_kernel(k_len, sigma)[k_len//2,:]

# find L(defined in article)

def L(N):
    L = np.zeros((N-1, N))
    
    i,j = np.indices(L.shape)
    
    L[i==j] = 1
    
    L[i==j-1] = -1

    return L

# find Dh(Defined in article)
# the matrix will be in a different format so it can multiply the pixel vector

def Dx(N):
    return sparse.kron(sparse.eye(N), L(N))

# find Dv(Defined in article)
# the matrix will be in a different format so it can multiply the pixel vector

def Dy(N):
    return sparse.kron(L(N), sparse.eye(N))

# you may change these values to more convinient ones

n = 9

sigma = 5

#  read an image, the image must have both dimensions equal

image = imread("batman_24.png")
# image = imread("batman_32.png")
# image = imread("batman_64.png")
# image = imread("batman_128.png")
# image = imread("batman_256.png")
# image = imread("batman_512.png")

image_grey = Turn_grey(image)

blurred_image = getBlurredImage(image_grey, n, sigma)

m = n

kernel1d = gaussian1d(n, sigma)

# this will get the image dimension, since its squared it doesent matter wich dimension you pick
# here we git the first dimension

N = np.array(image_grey).shape[0]

T = toeplitz(kernel1d, N)

A = col_mat(m, N, T).dot(row_mat(N, T))

# here we flatten the blurred image
# look at https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html

flatted_blurred_image = blurred_image.flatten()

def f1(x, A = A, b = flatted_blurred_image):
    return la.norm(b-A.dot(x))**2
    
def f2(x, A = A, b = flatted_blurred_image ):
    return 2.0*(A.T).dot(A.dot(x) - b)

# this funcion finds the x that minimizes the function for the vector x
# it uses broyden-fletcher-goldfarb-shanno's algorithm
# check https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

#let's check how long this will take

t1 = TicToc()

t1.tic() #start

optim_output_1 = optimize.minimize(f1, np.zeros(N**2), method='L-BFGS-B', jac=f2, options={'disp':True})

final_image_BFGS = optim_output_1['x']

t1.toc() #stop

# here we do the least square problem. again with broyden-fletcher-goldfarb-shanno's algorithm
# but now we analyze the smooth least square 
# we use here lambda = 10⁻10, this is done through test

l = 1e-10

def f3(x, A = A, b = flatted_blurred_image, l = l):
    return la.norm(b - A.dot(x))**2 + l*(la.norm(Dx(N).dot(x))**2 + la.norm(Dy(N).dot(x))**2)

def f4(x, A = A, b = flatted_blurred_image, l = l):
    return 2*((A.T).dot(A.dot(x) - b) +l*((Dx(N).T).dot(Dx(N).dot(x)) + (Dy(N).T).dot(Dy(N).dot(x))))

t2 = TicToc()

t2.tic() #start

optim_output_2 = optimize.minimize(lambda x: f3(x),
                                 np.zeros(N**2),
                                 method='L-BFGS-B',
                                 jac=lambda x: f4(x),
                                 options={'disp':True}
)

t2.toc() #stop

final_image_smooth_BFGS = optim_output_2['x']

# showing all images

plt.imshow(blurred_image, cmap="Greys_r")

plt.show()

printImage(final_image_BFGS,N)

printImage(final_image_smooth_BFGS,N)

# all our results until now are based on python's scipy library
# now we will show our implementation to find the best non_blurred_image

want_to_see_QR = True

if want_to_see_QR:

    t3 = TicToc()

    t3.tic()

    A_ = scipy.sparse.vstack((A, l*Dx(N), l*Dy(N))).todense()

    Q,R = la.qr(A_, mode = "complete")

    m = A.shape[0]

    n = A_.shape[1]

    Q1 = Q[0:m,0:n]

    R1 = R[0:n,0:n] 

    b = flatted_blurred_image

    aux = np.dot((Q1.T),b)

    x_ = np.dot(la.inv(R1),aux.T)

    x_ = x_.T

    t3.toc()

    printImage(x_,N)