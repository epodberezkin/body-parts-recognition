import numpy as np
cimport numpy as np
DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

from cython.parallel import prange

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def make_features(int [:,:] Dp, int [:,:,:] Fp, int ymin, int xmin, int ymax, int xmax):
    
    cdef int FEATURES = 200
    cdef int SIZE_X = 640
    cdef int SIZE_Y = 480
    cdef int dy,dx
    cdef int x,y
    cdef int u0,u1,v0,v1,U,V
    cdef int f
    
    cdef np.ndarray[DTYPE_t, ndim=2] X_image = np.zeros((SIZE_Y*SIZE_X,FEATURES),dtype=DTYPE)
    cdef int [:,:] Xp = X_image
    
    for f in prange(FEATURES,nogil=True):
                
        u0 = Fp[f,0,0] # y coord of 1st point
        u1 = Fp[f,0,1] # x coord of 1st point
        v0 = Fp[f,1,0] # y coord of 2nd point
        v1 = Fp[f,1,1] # x coord of 2nd point
        
        for y in range(ymin,ymax):
            for x in range(xmin,xmax):
                if Dp[y, x] != 0:
                    dy = u0 * 1000 // Dp[y, x]
                    dx = u1 * 1000 // Dp[y, x]
                    if(y + dy >= 0) and (y + dy < SIZE_Y) and (x + dx >= 0) and (x + dx < SIZE_X):
                        U = Dp[y + dy, x + dx]
                    else:
                        U = 8160    
                    dy = v0 * 1000 // Dp[y, x]
                    dx = v1 * 1000 // Dp[y, x]
                    if(y + dy >= 0) and (y + dy < SIZE_Y) and (x + dx >= 0) and (x + dx < SIZE_X):
                        V = Dp[y + dy, x + dx]
                    else:
                        V = 8160   
                    Xp[y*SIZE_X + x, f] = U - V           
    return Xp