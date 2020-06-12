import numpy as np
import numpy.matlib as mlib
def image_grid(nrows, ncols):

    if (np.mod(nrows, 2) == 0):
        m =  np.arange(np.ceil(-nrows / 2), np.floor( nrows/ 2))
    else:
        m = np.arange(np.ceil(-nrows / 2), np.floor( nrows / 2)+1)


    if (np.mod(ncols, 2) == 0):
        n = np.arange(np.ceil(-ncols / 2),np.floor( ncols / 2))
    else:
        n = np.arange(np.ceil(-ncols / 2), np.floor( ncols / 2) +1)

    return m.T,n

def radon_transform_matrix(M, N):
    W = np.sqrt(M**2 + N**2 )
    rho = (np.arange(W/10)*10)-W/2
    theta = np.arange(180)
    L = len(rho);
    P = len(theta);
    R = np.zeros( M*N)
    m, n = image_grid(M,N)
    for i in range(P):
        phi = theta[i] * np.pi / 180;
        if (phi >= np.pi / 4  and phi <= 3 * np.pi / 4):
            step = min(1 / np.sqrt(2), 1 / abs(np.tan(np.pi/2 - phi)))
            t = np.arange(-W, W, step = step)
            T = len(t);
            rhom = mlib.repmat(rho.flatten(), 1, T )
            tn = mlib.repmat(t.flatten().T, L, 1 )
            mline = (rhom - (tn * np.cos(phi)).reshape(rhom.shape))/np.sin(phi)
            mline = mline.reshape(tn.shape)
            j = 0
            while j < L:
                p = np.round(tn[j,:] - np.amin(n)).astype(np.int32)
                q = np.round(mline[j,:] - np.amin(m)).astype(np.int32)
                inds = np.logical_and(np.logical_and(0 <= p,p < N - 1),np.logical_and(q< M - 1 ,0 <= q))
                R[np.unique(np.ravel_multi_index((q[inds], p[inds]),(M,N)))]= 1
                yield R
                j += 1
        else:
            step = min(1 / np.sqrt(2), 1 / abs(np.tan(phi)))

            t =  np.arange(-W, W, step = step)
            T = len(t);
            rhon = mlib.repmat(rho.flatten(), T, 1 )
            tm = mlib.repmat(t.flatten().T, 1, L )
            nline = (rhon - (tm * np.sin(phi)).reshape(rhon.shape))/np.cos(phi)
            nline = nline.reshape(tm.shape)
            j = 0
            while j < L:
                p = np.round(nline[ :,j] - np.amin(n)).astype(np.int32)
                q = np.round(tm[:,j] - np.amin(m)).astype(np.int32)
                inds = np.logical_and(np.logical_and(0 <= p, p < N - 1), np.logical_and(q < M - 1, 0 <= q))
                R[np.unique(np.ravel_multi_index((q[inds], p[inds]), (M, N)))] = 1
                yield R
                j += 1



# R = radon_transform_matrix(3,4)
# for i in R:
#     print(i)

