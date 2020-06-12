from skimage.transform import radon, iradon, rotate
import numpy as np
import warnings
from time import time
from my_iradon import my_iradon

def radon_transform(C, X, thetas = None):
    if hasattr(C, '__call__'):
        # C is a function
        # op = Radon2D(taxis = np.arange(180),haxis= np.arange(200),pxaxis = np.linspace(0,255,num=5))
        return C(X, circle = True, theta= thetas)

    else:
        # C is a matrix
        return C * X


def inverse_radon_transform(C_T, X):
    if hasattr(C_T, '__call__'):
        # C is a function
        return C_T(X, output_size = X.shape[0], circle= True)

    else:
        # C is a matrix
        return C_T * X


def prox_cauchy(x, gamma, mu):
    p = gamma ** 2 + 2 * mu - (x * x / 3)
    q = x * gamma * gamma + (x ** 3 / 27) + (x / 3) * (gamma ** 2 + 2 * mu)
    DD = p ** 3 / 27 + q ** 2 / 4
    s = np.power(np.abs(q / 2 + np.sqrt(DD)), 1 / 3)*np.sign(q/2 + np.sqrt(DD))
    t = np.power(np.abs(q / 2 - np.sqrt(DD)), 1 / 3)*np.sign(q/2 + np.sqrt(DD))
    z = x / 3 + s + t
    return z


def cauchy_regularizer(x, gamma):
    return np.log(x * x + gamma * gamma) - np.log(gamma)


"""The forward backward algorithm. This function is generalized as possible
but still relied heavily on using a cauchy penalty as regularizer, by the
algorithm prox_cauchy.
The function receives the following arguments:
1)Y - the image.
2)max_iter - the maximal number of iterations to perform the algorithm.
3)mu = the step size.
4)gamma - the parameter for the cauchy distribution. Relevant only if cauchy 
regulatization is employed.
5) C - The inverse radon transform - A function handle or a matrix
6) C_T - The radon transform - a function handle or matrix.
7) regularizer - a function handle.
8)epsilon - a threshold to determine convergence and to stop iterations
early.
"""


def fb_algorithm(Y, max_iter=1, mu= 0.1 , gamma=None, thetas =None ,\
                 C=iradon, C_T= radon, regularizer=None, epsilon=0.01):
    # check validity of parmaeters
    if gamma is not None and gamma < (np.sqrt(mu) / 2):
        message = 'gamma and mu do not satisfy condition for convexity,aborting!'
        warnings.warn(message)
        return

    if gamma is None:
        gamma = (np.sqrt(mu) / 2)

    # initial guess, X is the radon transform of Y, unregularized
    X = radon_transform(C_T, Y, thetas)
    X /= np.amax(X)

    for i in range(max_iter):

        temp = inverse_radon_transform(C, X)

        temp -= Y
        u = X - mu * radon_transform(C_T, temp)
        u/=np.amax(u)
        X_curr = prox_cauchy(u, gamma, mu)
        #
        # if np.linalg.norm(X_curr - X) / np.linalg.norm(X) < epsilon:
        #     return X_curr
        X = X_curr

    return X
