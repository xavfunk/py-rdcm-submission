import numpy as np
import ctypes
from scipy.fft import fft, ifft


#def euler_integration(A,C,U,B,D,rho,alphainv,tau,gamma,kappa,params):
def euler_integration(A = np.array([[-1]]), C = np.vstack((np.ones((1,1)),np.zeros((1599,1)))), 
                      U = np.vstack((np.ones((1,1)),np.zeros((1599,1)))), B = np.array(0),
                      D = np.array(0), rho = np.array(.32), alphainv = np.array(3.125),
                      tau = np.array(2.0), gamma = np.array(.32),
                      kappa = np.array(.64), params = np.array([0.1250, 1600, 1, 1, 0, 1, 1])):

    """wrapping euler_integration.so"""
    
    # load shared library
    lib = ctypes.CDLL("./euler_integration.so")
    euler = lib.euler_inplace

    #arrange inputs
    A = A.T.flatten().astype(np.double)
    C = C.T.flatten().astype(np.double)
    U = U.T.flatten().astype(np.double)
    B = B.flatten().astype(np.double)
    D = D.flatten().astype(np.double)
    rho = rho.flatten().astype(np.double)
    alphainv = ctypes.c_double(alphainv)
    tau = tau.flatten().astype(np.double)
    gamma = ctypes.c_double(gamma)
    kappa = kappa.flatten().astype(np.double)
    params = params.flatten().astype(np.double)

    # A = A.T.astype(np.double)
    # C = C.astype(np.double)
    # U = U.astype(np.double)
    # B = B.T.astype(np.double)
    # D = D.T.astype(np.double)
    # rho = rho.astype(np.double)
    # alphainv = ctypes.c_double(alphainv)
    # tau = tau.astype(np.double)
    # gamma = ctypes.c_double(gamma)
    # kappa = kappa.astype(np.double)
    # params = params.astype(np.double)
    
    
    # initialize outputs
    x_out = np.zeros(int(params[1]*params[2])).astype(np.double)
    s_out = np.zeros(int(params[1]*params[2])).astype(np.double)
    f1_out = np.zeros(int(params[1]*params[2])).astype(np.double)
    v1_out = np.zeros(int(params[1]*params[2])).astype(np.double)
    q1_out = np.zeros(int(params[1]*params[2])).astype(np.double)

    
    # set input/output types
    types = [ctypes.c_void_p for i in range(16)]
    types[6] =  ctypes.c_double
    types[8] =  ctypes.c_double
    euler.argtypes = types
     
    euler(# inputs
            ctypes.c_void_p(A.ctypes.data),
            ctypes.c_void_p(C.ctypes.data),
            ctypes.c_void_p(U.ctypes.data),
            ctypes.c_void_p(B.ctypes.data),
            ctypes.c_void_p(D.ctypes.data),
            ctypes.c_void_p(rho.ctypes.data),
            alphainv,
            ctypes.c_void_p(tau.ctypes.data),
            gamma,
            ctypes.c_void_p(kappa.ctypes.data),
            ctypes.c_void_p(params.ctypes.data),
          # outputs
            ctypes.c_void_p(x_out.ctypes.data),
            ctypes.c_void_p(s_out.ctypes.data),
            ctypes.c_void_p(f1_out.ctypes.data),
            ctypes.c_void_p(v1_out.ctypes.data),
            ctypes.c_void_p(q1_out.ctypes.data))
    
    if len(A) == 1: # 1 region case
        return x_out, s_out, f1_out, v1_out, q1_out
    
    else: # n region case: reshape output
        return (x_out.reshape(int(params[2]), int(params[1])),
            s_out.reshape(int(params[2]), int(params[1])),
            f1_out.reshape(int(params[2]), int(params[1])), 
            v1_out.reshape(int(params[2]), int(params[1])), 
            q1_out.reshape(int(params[2]), int(params[1])))

def reshape_output(arr, params):
    """helper function to reshape, depreacated"""
    return arr.reshape(int(params[2]), int(params[1]))

if __name__ == "__main__":
    
    """testing on HRF"""
    
    from quick_BOLD import quick_bold
    import scipy.io as sio
    import time

    nr = 1 # number regions

    # load inputs
    ie = sio.loadmat('test_inputs/euler_integration_inputs_HRF')['inputs_euler']
    A = ie[0][0][0]
    C = ie[0][0][1]
    U = ie[0][0][2]
    mArrayB = ie[0][0][3]
    mArrayD = ie[0][0][4]
    rho = ie[0][0][5]    
    alphainv = ie[0][0][6][0]
    tau = ie[0][0][7]
    gamma = ie[0][0][8][0]
    kappa = ie[0][0][9]
    paramList = ie[0][0][10][0]

    paramList[5] = 0.0
    paramList[6] = 0.0
    
    start = time.time()

    x, s, f, v, q = euler_integration(A = A, C = C, U = U,
                                      B = mArrayB, D = mArrayD,
                                      rho = rho, alphainv = alphainv,
                                      tau = tau, gamma = gamma,
                                      kappa = kappa, params = paramList)
    end = time.time()
    print('Time taken in seconds -', end - start)
    
    
    h = quick_bold(x, s, f, v, q, nr)[0].flatten()
    
    
    
    """testing on generation"""
    
    import scipy.io as sio
    import time

    nr = 50 # number of regions
    
    # load inputs
    ieg = sio.loadmat('test_inputs/inputs_euler_gen')['inputs_euler_gen']
    A = ieg[0][0][0]#.T
    C = ieg[0][0][1]#.T
    U = ieg[0][0][2]#.T
    mArrayB = ieg[0][0][3]
    mArrayD = ieg[0][0][4]
    rho = ieg[0][0][5]    
    alphainv = ieg[0][0][6][0]
    tau = ieg[0][0][7]
    gamma = ieg[0][0][8][0]
    kappa = ieg[0][0][9]
    params = ieg[0][0][10][0]
    params[5] = 0.0
    params[6] = 0.0
    
    start = time.time()

    x, s, f, v, q = euler_integration(A = A, C = C, U = U,
#                                      B = mArrayB, D = mArrayD,
                                      B = np.array(0), D = np.array(0),

                                      rho = rho, alphainv = alphainv,
                                      tau = tau, gamma = gamma,
                                      kappa = kappa, params = params)
    end = time.time()
    print('Time taken in seconds -', end - start)
    
    #h_, x = quick_bold(x, s, f, reshape_output(v), reshape_output(q), nr)
    
    x_ = reshape_output(x, params = params)
    N = 130272
    y = np.zeros((N, nr))
                 
    # convolving signal with HRF
    # ignored stacking effects
    # for i in range(nr):
    #     y[:,i] = ifft(fft(x[:,i]) * fft(np.hstack((h, np.zeros(N-len(h))))))
        
        
     