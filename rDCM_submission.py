# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:55:45 2021

@author: Xaver Funk

This file contains the rDCM class definition that can perform regression DCM,
as well as two testcases on simulated and real data, respectively
"""

import numpy as np
import scipy.special as sp
from scipy.fft import fft, ifft
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
from wrap_euler import euler_integration
from tqdm import tqdm



class DCM:
    
    def __init__(self, Y = None, U = None, a = None,  b = None, c = None, d = None,
                  Tp = None, **kwargs):
            
            """
            initializes a DCM object
            
            Y                   - observed or simulated data (see class Y)
            U                   - experimental input (see class U)
            a (nr x nr)         - prior on A: connectivity between regions
            b (nr x nr x nu)    - prior on B: modulatory influence of inputs
            c (nr x nu)         - prior on C: direct influence of inputs
            d (nr x nr x nr)    - prior on D: nonlinear effects
            Tp                  - True parameters for synthetic DCM            
            """
        
            self.Y = Y
            self.U = U
            self.Tp = Tp
            
            # parse kwargs for additional attributes
            self.__dict__.update(kwargs)
            
            try:
                self.nr = self.Y.y[1] # number of regions
            except AttributeError:
                try:
                    self.nr = nr
                except NameError:
                    print('No information on number of regions detected. Must specify either Y or nr')
                
            try:
                self.v = self.Y.y[0] # number of datapoints      
            except AttributeError:
                try:
                    self.v = v
                except NameError:
                    print('No information on number of datapoints detected. Must specify either Y or v')
            
            try:
                self.names = Y.names
            except AttributeError:
                self.names = None
            
            # check if Y is specified
            if Y is not None:
                
                #set default connectivity matrix -> full
                if isinstance(a, np.ndarray):
                    self.a = a
                else:
                    self.a = np.ones((Y.y.shape[1], Y.y.shape[1])) #fully connected
                
                if isinstance(b, np.ndarray):
                    self.b = b
                else:
                    self.b = np.zeros((Y.y.shape[1], Y.y.shape[1], U.u.shape[1])) #fully connected
            
                if isinstance(c, np.ndarray):
                    self.c = c
                else:
                    self.c = np.ones((Y.y.shape[1], U.u.shape[1])) #fully connected
            
                if isinstance(a, np.ndarray):
                    self.d = d
                else:
                    self.d = np.zeros((Y.y.shape[1], Y.y.shape[1], Y.y.shape[1])) #fully connected
            
                
                
                if U is None:
                    print("\nNOTE: No inputs specified! Assuming resting-state model... \n")
                
                    # these are dummy variables for rs
                    self.b = np.zeros((1,Y.y.shape[1],Y.y.shape[1]))
                    self.c = np.zeros((Y.y.shape[1],1))
                    self.d = np.zeros((1,Y.y.shape[1],Y.y.shape[1]))
               
                
                else:
                   
                    # number of inputs
                    self.nu = U.u.shape[1]
                                      
                    # specify inputs
                    # adjust dimensionality if U.dt != Y.dt*16 (microtime resolution)
                    if U.u.shape[0] == self.Y.y.shape[0]*16:
                        self.U = U
                    
                    elif U.u.shape[0] == self.Y.y.shape[0]:
                        try:
                            self.U.u = np.repeat(U.u, 16, axis = 0)
                        except np.AxisError:
                            self.U.u = np.repeat(U.u, 16)
                    else:
                        print('\nERROR: Dimensionality of data (y) and inputs (u) does not match! \n')
                        print('Please double-check... \n')
                        return
                    
            else:

                print('No Y specified, assuming synthetic data will be generated')

                if isinstance(a, np.ndarray):
                    self.a = a
                else:
                    self.a = np.ones((Y.y.shape[1], Y.y.shape[1])) #fully connected
                
                # specify default/dummy matrices b to d
                if isinstance(b, np.ndarray):
                    self.b = b
                else:
                    self.b = np.zeros((U.u.shape[1], Y.y.shape[1], Y.y.shape[1]))
                
                if isinstance(c, np.ndarray):
                    self.c = c
                else:
                    self.c = np.ones((Y.y.shape[1], U.u.shape[1])) # default to full input
                    
                if isinstance(d, np.ndarray):
                    self.d = d
                else:
                    self.d = np.zeros((1,Y.y.shape[1],Y.y.shape[1]))
                    
                    
        
    def get_prior(self):
        
        """
        returns and stores priors on model parameters (theta) and noise precision
        (tau). 
        
        calls:
            spm_dcm_fmri_priors
        
        output:
            m0      - prior mean on theta
            l0      - prior covariance on theta
            a0      - prior shape parameter on tau
            b0      - prior rate parameter on tau
        
        """
        
        # get n of regions and input
        (nr, nu) = self.c.shape    
    
        # get priors on model parameters
        pE, pC = spm_dcm_fmri_priors(self.a, self.b, self.c, self.d)
        
        # set prior mean of A to 0, except for self-connections
        pE.A = np.zeros(pE.A.shape) + np.diag(np.diag(pE.A))
        
        # specify the prior mean on theta
        m0 = np.hstack((pE.A, pE.B.reshape(nr, nr*nu), pE.C))
        
        # prior precision: inverse of prior covariance
        pC.A[pC.A.nonzero()] = 1.0/pC.A[pC.A.nonzero()]
        pC.B[pC.B.nonzero()] = 1.0/pC.B[pC.B.nonzero()]
        pC.C[pC.C.nonzero()] = 1.0/pC.C[pC.C.nonzero()]
        pC.D[pC.D.nonzero()] = 1.0/pC.D[pC.D.nonzero()]
        pC.transit = 1.0/pC.transit
        pC.decay = 1.0/pC.decay
        pC.epsilon = 1.0/pC.epsilon
        
        # prior precision of baseline
        if np.any(pC.C[:,-1]):
            pC.C[:,-1] = 10e-8
        
        # prior precision on theta
        l0 = np.hstack((pC.A, pC.B.reshape(nr, nr*nu), pC.C))
        
        # setting priors on noise precision
        a0 = np.array([2.0])
        b0 = np.array([1.0])
        
        # store priors
        self.priors = Bunch(m0 = m0, l0 = l0, a0 = a0, b0 = b0)
        
        # return priors
        return [m0, l0, a0, b0]    
    
    
    def set_options(self, type, sparse = False, **kwargs):
        
        """
        intializes, sets and returns options
        
        type        - type of DCM, empirical or synthetic
        sparse      - flag for sparse rDCM, not implemented yet
        **kwargs    - different option settings
        """

        # init options as a sklearn.Bunch
        options = Bunch()
        options['type'] = type
        
        # defaults for synthetic DCM
        if type == 's':
            options['SNR'] = 3
            options['y_dt'] = 0.5
        
        # defaults for empirical DCM
        else:
            options['y_dt'] = self.Y.dt
        
        # defaults for both empirical and synthetic DCMs
        options['bilinear'] = 0
        options['u_shift'] = 0
        options['filter_str'] = 1
        options['coef'] = 1
        options['visualize'] = 1
        options['compute_signal'] = 1
        options['evalCp'] = 0

        # overwrite defaults with **kwargs
        for key, value in kwargs.items():
            options[key] = value


        # check if h is already specified
        if 'h' in options:
            
            # add options to self
            self.options = options
        
            # return options
            return options
        
        else:
            # set convolution parameters
            options['conv_dt'] = self.U.dt 
            options['conv_length'] = self.U.u.shape[0]
            options['conv_full'] = True
            for key, value in kwargs.items():
                options[key] = value
            
            # get convolution HRF
            options['h'] = self.get_convolution(options) 
                
            # add options to self
            self.options = options
        
            # return options 
            return options
        

    def euler_gen(self, Ep = None, y_dt = 0.5):
        
        """
        calculates BOLD signal for specified parameters with fixed HRF kernel
        defaults to simple HRF with one input event at U[0] when Ep = None
        Ep - parameters A,B,C,D,transit,decay,epsilon
        """
        # get y_dt
        try:
            y_dt = self.Y.dt
        except AttributeError:
            y_dt = y_dt
        
        if Ep is None:
            
            # using defaults for simple HRF 
            C = np.vstack((np.ones((1,1)),np.zeros((self.U.u.shape[0]-1,1))))
            U = np.vstack((np.ones((1,1)),np.zeros((self.U.u.shape[0]-1,1))))
            params = np.array([self.U.dt, self.U.u.shape[0], 1, 1, 0, 1, 1])
            
            # call euler_integration with defaults
            x, s, f, v, q = euler_integration(C = C, U = U, params = params)
            epsilon = np.array([1])
            
            # length of input
            L = self.U.u.shape[0]
            
            # number of regions
            nr = 1
            
            # for simple HRF, use all idxs
            idxs = np.arange(L)  
        
        else:
            # driving inputs
            nr = Ep['A'].shape[0]
            C = self.U.u @ Ep.C.T/16

            # hemodynamic constants
            H = (0.64, 0.32, 2.00, 0.32, 0.32)
            
            # constants for hemodynamic model
            rho                      = 0.32*np.ones((nr, 1)) # used to be called oxygenExtractionFraction 
            alphainv                 = 1/H[3]
            tau                      = H[2]*np.exp(Ep['transit']) 
            gamma                    = H[1]
            kappa                    = H[0]*np.exp(Ep['decay']) 
            epsilon                  = np.exp(Ep['epsilon'])*np.ones((1, nr))
         
            params = np.array([self.U.dt, self.U.u.shape[0], 
                               nr, self.U.u.shape[1], 0, 0, 0])
            
            # call euler_integration
            x, s, f, v, q = euler_integration(Ep['A'].T, C, self.U.u, B = np.array(0), 
                                              D = np.array(0), rho=rho, alphainv = alphainv,
                                              tau = tau, gamma = gamma, kappa = kappa, params = params)
            
            # length of input
            L = self.U.u.shape[0]
                
            # number of regions
            nr = self.nr
            
            # get the indices that coincide with the data timepoints
            idxs = np.arange(L)[::int(np.floor(y_dt/self.U.dt))]


        # constants for BOLD signal equation
        relaxationRateSlope      = 25.0
        frequencyOffset          = 40.3
        oxygenExtractionFraction = 0.4 * np.ones((1, nr))
        echoTime                 = 0.04
        restingVenousVolume      = 4.0
    
        # coefficients of BOLD signal equation
        coefficientK1  = 4.3 * frequencyOffset * echoTime * oxygenExtractionFraction
        coefficientK2  = epsilon * (relaxationRateSlope * oxygenExtractionFraction * echoTime)
        coefficientK3  = 1.0 - epsilon
    
        # calculate BOLD signal     
        if nr == 1:
            # for 1 region
            y = restingVenousVolume * (coefficientK1.T * (1-q[idxs]) + coefficientK2.T * (1-q[idxs]/v[idxs]) + coefficientK3.T * (1-v[idxs]))
        else:
            # for n regions
            y = restingVenousVolume * (coefficientK1.T * (1-q[:,idxs]) + coefficientK2.T * (1-q[:,idxs]/v[:,idxs]) + coefficientK3.T * (1-v[:,idxs]))

        return y, x
    
    def get_convolution(self, options = None):
        """
        calls euler_gen with defaults to create convolution HRF
        """
        
        if options == None:
            options == self.options
        
        if options['conv_full'] == True:
            r_dt = 1
        else:
            r_dt = self.U.u.shape[0]//options['conv_length'] # this is also 1 with defaults
        
        # create HRF
        y = self.euler_gen()[0].T
        
        # sample HRF [start:stop:step]
        h = y[::r_dt]
               
        return h

    
    def create_regressors(self, options = None):
    
        """
        outputs:
            X   - design matrix containing data, bilinear terms, inputs convolved
                  with HRF; all in frequency domain
            Y   - temporal derivative of data in frequency domain
            
            note that bilinear terms are not yet supported and consequently set to 0
        """
        
        if options == None:
            options = self.options
    
        # circular shift of the stimulus input function if u_shift is specified
        self.U.u = np.roll(self.U.u, options["u_shift"], 0)
    
        # unwrapping DCM
        y        = self.Y.y
        u        = self.U.u
        u_dt	 = self.U.dt
        y_dt     = self.Y.dt
        r_dt     = int(y_dt // u_dt)
        Nu, nu = u.shape
        Ny, nr = y.shape
    
    
        ## create regressors
    
        # fourier transform of hrf
        h_fft = fft(options['h'].flatten())  
        # fourier transform of BOLD signal
        y_fft = fft(y, axis = 0)
        
        # convolution of stimulus input and hrf
        u = ifft(fft(u, axis = 0) * np.tile(h_fft, (nu, 1)).T, axis = 0).real
    
        # get confounds, if empty: set to constant if task, set to 0 if rest        
        if self.U.X0 is None:
            
            # check for resting-state
            if np.any(self.U.u): # inputs provided -> no rs
                self.U.X0 = np.ones((self.U.u.shape[0], 1)) # constant confound
                options['filtu'] = 1
            
            else: # resting state
                self.U.X0 = np.zeros((self.U.u.shape[0],1)) # empty confound
                options['filtu'] = 0
            
        # add confounds (e.g. constant, linear trend, sinusoids) as additional columns      
        u = np.hstack((u, self.U.X0))
        
        # make u_fft
        if r_dt > 1:
            u = u[::r_dt]
            h_fft = fft(options['h'][::r_dt], axis = 0) 
            
        u_fft = fft(u, axis = 0)
        
        # filter the signal
        if options['filter_str'] != 0:
            y_fft, idx = filter_signal(y_fft, u_fft, h_fft, Ny, filtu = options['filtu'], filter_str = options['filter_str'])
        else:
            idx = np.ones(y_fft.shape, dtype = bool)

        # coefficients to transform fourier of the function to the fourier of the derivative
        coef = np.exp(2*np.pi*1j*np.arange(Ny).reshape(Ny,1)/Ny)-1
        
        # calculate derivative of y_fft
        yd_fft = np.repeat(coef, nr, axis=1) * y_fft /y_dt
        
        # remove filtered out frequencies
        yd_fft[~idx] = np.nan
        
        # bilinear term, for now all 0s until supported
        yu_fft = np.zeros((Ny, nr * (nu + self.U.X0.shape[1])))        
        
        ## combine regressors
        
        # create X matrix
        if options['type'] == 'r':
            X = np.hstack((y_fft, yu_fft, u_fft))
            
        elif 'scale_u' in options:
            X = np.hstack((y_fft, yu_fft, u_fft))
        
        else:
            X = np.hstack((y_fft, yu_fft, u_fft/r_dt))

        """
        # create X matrix without bilinear term
        if options['type'] == 'r':
            X = np.hstack((y_fft, u_fft))
            
        elif 'scale_u' in options:
            X = np.hstack((y_fft, u_fft))
        
        else:
            X = np.hstack((y_fft, u_fft/r_dt))
        """
        
        # create Y matrix
        Y = yd_fft
        Y = reduce_zeros(X, Y)
        
        return X, Y
    
    
    def ridge(self, X, Y, evalCp = False):
        """
        Variational Bayesian inversion of linear DCM with rDCM
        implements VB update equations from Frässle et al. 2017
        
        Input:
            X - design matrix
            Y - data: derivative of BOLD timeseries in frequency domain
        
        Output:
            self.posterior  - contains posterior estimates
            self.signal     - contains original and predicted temporal derivative 
                                in frequency domain
        """
                
        # precision limit
        pr = 10e-5    
        
        # r_dt
        r_dt = int(self.Y.dt//self.U.dt)
        
        # add confound regressor dimensions
        Nc = self.U.X0.shape[1] # second element of shape of X0 gives number of confounds
        
        for nc in range(Nc):
            self.b = np.dstack((self.b, self.b[:,:,0]))
            self.c = np.hstack((self.c, np.ones((self.c.shape[0], 1))))
        
        # reshape b
        self.b = np.swapaxes(self.b, 1, 2)
        
        # get number of regions and inputs
        nr, nu = self.c.shape
              
        # no baseline regressor for simulations
        if (self.options.type == 's'):
            self.c[:,-1] = 0
      
        # define relevant params and get priors
        idx = np.where(np.hstack((self.a, self.b.reshape(nr, nr*nu), self.c))>0, True, False)
        m0, l0, a0, b0 = self.get_prior()
      
        # define results arrays
        mN = np.zeros(idx.shape)
        sN = [] 
        aN = np.zeros((nr, 1))
        bN = np.zeros((nr, 1))
        logF = np.zeros((1, nr))
        
        # define array for predicted derivative of signal (in freq domain)
        yd_pred_rdcm_fft = np.zeros(self.Y.y.shape, dtype = complex)
        yd_pred_rdcm_fft[:] = np.NaN
              
        # array to store free energies
        logF = np.zeros(nr, dtype = complex)
        
        # array to store free energy trajectories
        F_trajectory = [[] for i in range(nr)]
        
        log_lik_trajectory = [[] for i in range(nr)]  
        log_p_prec_trajectory = [[] for i in range(nr)]
        log_p_weight_trajectory = [[] for i in range(nr)]
        log_q_prec_trajectory = [[] for i in range(nr)]
        log_q_weight_trajectory = [[] for i in range(nr)]  
                
        # iterate over region
        for k in tqdm(range(nr), desc="iterating over regions"):
      
              # find finite values
              idx_y = np.logical_not(np.isnan(Y[:,k]))
              
              # prepare regressors and data by removing unnecessary dimensions
              X_r = X[idx_y]
              X_r = X_r[:, idx[k,:]]
              Y_r = Y[idx_y, k]
              
              # effective number of datapoints
              N_eff = sum(idx_y)/r_dt

              # effective dimensionality -> number connections entering region
              D_r = sum(idx[k,:])

              ## read priors
          
              # prior covariance matrix
              l0_r = np.diag(l0[k, idx[k, :]])
          
              # prior means
              m0_r = m0[k, idx[k,:]].T
          
              # compute X'X and X'Y
              # NOTE .T is not equal to matlab ' for complex numbers, therefore, added .conjugate()
              W = X_r.conjugate().T @ X_r
              v = X_r.conjugate().T @ Y_r
          
              ## compute optimal theta and tau
              
              # initialize tau
              t = a0/b0

              # estimate alpha_N
              aN_r = a0 + N_eff / (2*r_dt)
          
              # cycle stops after 500 iterations
              count = 500
          
              # set old F
              logF_old = np.inf
          
              # convergence criterion
              convergence = 0

              while not convergence:
    			
                  # update posterior covariance matrix
                  sN_r = np.linalg.inv(t * W + l0_r)
              
                  # update posterior means
                  mN_r = sN_r@(t*v + l0_r @ m0_r)
              
                  # update posterior rate parameter
                  QF = ((Y_r-X_r@mN_r).conjugate().T @ (Y_r - X_r @ mN_r)/2 + np.trace(W @ sN_r)/2)
                  bN_r = b0 + QF

                  # update tau
                  t = aN_r/bN_r
                  
                  ## compute model evidence
                  
                  # compute components of the model evidence
                  log_lik      = N_eff*(sp.digamma(aN_r) - np.log(bN_r))/2 - N_eff*np.log(2*np.pi)/2 - QF*t
                  log_p_weight = (-1/2*np.linalg.slogdet(l0_r)[1] - D_r*np.log(2*np.pi)/2 - (mN_r-m0_r).T @ l0_r @ (mN_r-m0_r)/2 - np.trace(l0_r*sN_r)/2)
                  log_p_prec   = a0*np.log(b0) - sp.gammaln(a0) + (a0-1)*(sp.digamma(aN_r) - np.log(bN_r)) - b0*t
                  log_q_weight = 1/2*np.linalg.slogdet(sN_r)[1] + D_r*(1+np.log(2*np.pi))/2
                  log_q_prec   = aN_r - np.log(bN_r) + sp.gammaln(aN_r) + (1-aN_r)*sp.digamma(aN_r)

                  # compute the negative free energy per region
                  logF[k] = (log_lik + log_p_prec + log_p_weight + log_q_prec + log_q_weight)[0]
                
                  # check whether convergence is reached
                  if  ((logF_old - logF[k].real)**2 < pr**2):
                        convergence = 1
              
                  # store old negative free energy
                  logF_old = logF[k].real
                  
                  F_trajectory[k].append(logF_old)

                  log_lik_trajectory[k].append(log_lik)
                  log_p_prec_trajectory[k].append(log_p_prec)
                  log_p_weight_trajectory[k].append(log_p_weight)
                  log_q_prec_trajectory[k].append(log_q_prec)
                  log_q_weight_trajectory[k].append(log_q_weight)
                  
                  
                  # decease the counter
                  count -= 1
              
                  # end optimization when number of iterations is reached
                  if count<0:
                        break     

              ## re-compute model evidence
              
              # expected log-likelihood
              log_lik      = N_eff*(sp.digamma(aN_r) - np.log(bN_r))/2 - N_eff*np.log(2*np.pi)/2 - QF*t
              
              # expected ln p(theta)
              #log_p_weight = (1/2*np.linalg.slogdet(l0_r)[1] - D_r*np.log(2*np.pi)/2 - (mN_r-m0_r).T @ l0_r @ (mN_r-m0_r)/2 - np.trace(l0_r*sN_r)/2)#.real
              log_p_weight = (-1/2*np.linalg.slogdet(l0_r)[1] - D_r*np.log(2*np.pi)/2 - (mN_r-m0_r).T @ l0_r @ (mN_r-m0_r)/2 - np.trace(l0_r*sN_r)/2)#.real
              
              # expected ln p(tau)
              log_p_prec   = a0*np.log(b0) - sp.gammaln(a0) + (a0-1)*(sp.digamma(aN_r) - np.log(bN_r)) - b0*t
              
              # expected ln q(theta)
              log_q_weight = 1/2*np.linalg.slogdet(sN_r)[1] + D_r*(1+np.log(2*np.pi))/2
              
              # expected ln q(theta)
              #log_q_prec   = aN_r - np.log(bN_r) + sp.gammaln(aN_r) + (1-aN_r)*sp.digamma(aN_r)
              log_q_prec   = aN_r - np.log(bN_r) + sp.gammaln(aN_r) - (aN_r-1)*sp.digamma(aN_r)

              # region-specific negative free energy
              logF[k]      = (log_lik + log_p_prec + log_p_weight + log_q_prec + log_q_weight)[0]

              # store region-specific parameters
              mN[k, idx[k,:]] = mN_r.real
              sN.append(sN_r)
              aN[k] = aN_r.real
              bN[k] = bN_r.real

              # get the predicted signal from the GLM (in frequency domain) (derivative of y)
              yd_pred_rdcm_fft[:,k]	= X[:,idx[k,:].T] @ mN_r

        # save results      
        self.posterior = Bunch(mN = mN, aN = aN, bN = bN, sN = sN, logF_r = logF.real, idx = idx)
        
        # store parameters
        
        # remove confound regressors
        self.b = np.delete(self.b, np.s_[-Nc:], 1)
        self.c = np.delete(self.c, np.s_[-Nc:], 1)
        
        # update nr, nu after removing confounds
        nr, nu = self.c.shape
        
        # desentangling mN -> means on connectivity parameters
        self.posterior['A'] = mN[:,:nr]
        self.posterior['B'] = mN[:nr, nr:nr+nr*nu].reshape(nr, nr, nu)
        self.posterior['C'] = mN[:nr, -(nu+Nc):-Nc]
        self.posterior['baseline'] = mN[:nr, -Nc :] # holds confound regressors
        
        # modify driving inputs
        if self.options.type == 'r':
            self.posterior.C = self.posterior.C * 16
        
        # free energy trajectories
        self.F_trajectory = F_trajectory
        self.log_lik_trajectory = log_lik_trajectory
        self.log_p_prec_trajectory = log_p_prec_trajectory
        self.log_p_weight_trajectory = log_p_weight_trajectory
        self.log_q_prec_trajectory = log_q_prec_trajectory
        self.log_q_weight_trajectory = log_q_weight_trajectory  
        
        # store precision parameters
        self.posterior.t = aN/bN
        
        ## store free energy
        self.posterior.logF = np.sum(logF.real) # total        
        
        # store original and predicted temporal derivative in freq domain
        self.signal = Bunch(yd_source_fft = Y, yd_pred_rdcm_fft = yd_pred_rdcm_fft )
        
        return self.posterior, self.signal
             
    
    def compute_statistics(self):
        """
        Compares the estimated parameters from the posterior to a ground truth,
        (Tp - True parameters), if applicable.
        This is done in terms of MSE, normalized MSE and sign errors
        """

        if self.Tp is not None:
    
            # get posterior estimates of A, B, C
            posterior = np.hstack((self.posterior.A.flatten(),
                                   self.posterior.B.flatten(),
                                   self.posterior.C.flatten()))
            
            # get true (or VBL) params of A, B, C
            ground_truth = np.hstack((self.Tp.A.flatten(),
                                      self.Tp.B.flatten(),
                                      self.Tp.C.flatten()))
            
            # compute stats
            self.statistics = Bunch()
            self.statistics.mse = np.mean((posterior - ground_truth)**2)
            self.statistics.mse_n = self.statistics.mse/np.linalg.norm(ground_truth) # normalized mse
            self.statistics.sign_errors = np.sum(posterior * ground_truth < 0) # counts negative values -> sign errors
        
        else:
            
            self.statistics = Bunch()
            self.statistics.mse = None
            self.statistics.mse_n = None # normalized mse
            self.statistics.sign_errors = None

        # compute signal: call compute_signals        
        self.compute_signals()
        
    
    def estimate(self, set_options = True, **kwargs):
        """
        wrapping set_options, create_regressors, ridge, compute_statistics
        for intuitive API-calls.
        
        **kwargs will be passed to set_options for putative option-setting
        """
        
        # set options
        if set_options == True:
            self.set_options(**kwargs)
        
        # create the regressors
        X, Y = self.create_regressors(self.options)
        
        # learn parameters
        self.ridge(X, Y)
        
        # compute statistics
        self.compute_statistics()
        
        return self.posterior, self.signal, self.statistics, self.residuals
        
    
    def compute_signals(self):
        """
        if ground truth is available ie synthetic model:
        (1) calculates mse of the clean synthetic ground truth signal wrt noisy ground truth
        (2) calculates mse of signal simulated with learned parameters wrt noisy ground truth
        
        if the model is empirical:
            calculates mse of signal simulated with learned parameters wrt observed signal
        """

        # get true or measured signal
        self.signal.y_source = self.Y.y
        
        if self.options.type == 's':
            # generate and save a noise-free signal with Tp
            self.signal.y_clean = self.generate(SNR = np.inf, y_dt = self.Y.dt, save_y = False)[0]      
            
            # calculate MSE of the noisy data in self.Y.y
            self.residuals = Bunch()
            self.residuals.y_mse_clean = np.mean((self.signal.y_source - self.signal.y_clean)**2)
                
        # set up Ep to feed generate
        Ep = Bunch(A = self.posterior.A,
                   B = self.posterior.B, 
                   C = self.posterior.C,
                   baseline = self.posterior.baseline,
                   transit = np.zeros(self.nr),
                   decay = np.zeros(self.nr),
                   epsilon = np.array([0]))
                
        # generate predicted signal -> call generate
        self.signal.y_pred_rdcm = self.generate(SNR = np.inf, y_dt = self.Y.dt, save_y = False, Ep = Ep)[0]
        
        # add confounds to time series
        self.signal.y_pred_rdcm += (Ep.baseline @ self.U.X0.T[:,::int(self.Y.dt/self.U.dt)]).T
        
        # compute MSE of predicted signal
        if self.options.type == 's':
            self.residuals.y_mse_rdcm = np.mean((self.signal.y_source-self.signal.y_pred_rdcm)**2) 
            self.residuals.R_rdcm = self.signal.y_source - self.signal.y_pred_rdcm 
        
        else:
            self.residuals = None
    

    def generate(self, SNR, y_dt = 0.5, save_y = True, Ep = None):
        """
        generates a BOLD signal from parameters Ep, defaults to self.Tp if no
        parameters were given.
        
        Input:
            SNR     - signal-to-noise ratio
            y_dt    - timestep for y
            save_y  - if True, saves the generated data as Y object
            Ep      - parameters A,B,C,D,baseline,transit,decay,epsilon
                        for signal generation
        Outputs:
            y       - clean generated BOLD signal
            y_noise - noisy generated BOLD signal
            x       - generated neuronal signal
        """
        
        if Ep == None:
            Ep = self.Tp
        
        try: # in case Y is already specified
            r_dt = int(self.Y.dt//self.U.dt)
            y_dt = self.Y.dt
        
        except AttributeError: # otherwise, use optional y_dt
            r_dt = int(y_dt // self.U.dt)
        
        N = self.U.u.shape[0]
        nr = self.nr      

        # specify array for the data: datapoints x regions
        y = np.zeros((N, nr))
        
        # if there is no HRF yet: call get_convolution with default params to get hrf
        try: 
            self.options.h
        except:
            self.get_convolution()
        
        # getting neuronal signal: call euler_gen
        x = self.euler_gen(Ep)[1].T
        
        # convolve neuronal signal with HRF
        for i in range(nr):        
            y[:,i]  = ifft(fft(x[:,i]) * fft(self.options.h.flatten())).real

        # sample
        y = y[np.round(np.arange(0,len(y),r_dt)).astype(int)]

        # add noise
        epsilon = np.random.normal(size = y.shape) @ np.diag(np.std(y, axis = 0, ddof = 1)/SNR)
        y_noise = y + epsilon
        
        # save generated data
        if save_y == True:
            
            self.Y = Y(y_noise, dt = y_dt)
            self.Y.y_clean = y 
            self.Y.x = x
            
        return y, y_noise, x
    
    
## auxiliary classes

class Y:
    
    """
    instantiates data object
    
    Input:
        y (Ny x nr) - BOLD time series
        dt          - TR, or data timestep
        names       - region names
    """
    
    def __init__(self, y, dt, names = None):
        
        self.y = y
        self.dt = dt
        if names:
            self.names = names
        else:
            self.names = ['region' + str(i) for i in range(self.y.shape[1])]

class U:
    
    """
    instantiates input object
    
    Input:
        u ((16*Ny) x nu )   - experimental input (16 for microtime resolution,
                              will be adapted if shape = N x U)
        names               - input names
        dt                  - timestep
        X0                  - confound regressors    
    """
    
    def __init__(self, u, dt, X0 = None, names = None):
        
        self.u = u
        self.dt = dt # Y.dt/16 # fix this later
        self.X0 = X0
        if X0 is not None:
            if len(X0.shape) == 1:
            	self.X0 = X0.reshape(len(X0), 1)
        else:
            self.X0 = X0
            
        if names:
            self.names = names
        else:
            self.names = ['input' + str(i) for i in range(self.u.shape[1])]


## auxiliary functions not dependent on DCM

def spm_dcm_fmri_priors(A, B, C, D = None):
    
    """
    Returns priors on expectation (mean) and covariance for DCM 
    
    Input:
        A, B, C, D priors on connectivity matrices
        
    output:
        pE - prior expectations (means)
        pC - prior covariances
        
    Note: the original version contains logic for two-state and endogenous fluct
    """

    nr = len(A)
    
    if D is None:
        D = np.zeros((nr, nr))
        
    # priors on states and shrinkage on A
    a = 8
    
    # enforce self-inhibition
    A = A > 0
    np.fill_diagonal(A, 0)
    
    # prior expectations
    # pE = Priors()
    pE = Bunch()
    pE.A = A /(64*nr) - np.eye(nr,nr)/2
    pE.B = B * 0
    pE.C = C * 0
    pE.D = D * 0
    
    # prior covariances
    pC = Bunch()
    pC.A = A * a/nr + np.eye(nr,nr)/(8*nr)
    pC.B = B
    pC.C = C
    pC.D = D
    
    # add hemodynamic priors
    pE.transit = np.zeros((nr,1))
    pE.decay = np.zeros((nr,1))
    pE.epsilon = np.zeros((1,1))
    
    pC.transit = np.zeros((nr,1)) + np.exp(-6)
    pC.decay = np.zeros((nr,1)) + np.exp(-6)
    pC.epsilon = np.zeros((1,1)) + np.exp(-6)
    
    return pE, pC



def filter_signal(y_fft, u_fft, h_fft, Ny, filtu = 1, filter_str = 1):
    """
    specifies informative frequencies and filters the fourier-transformed signal.
    Takes out high frequencies and frequencies with small power that are probably noise. 
    
    Input:
        y_fft       - fourier-transformed signal
        u_fft       - fourier-transformed 
        h_fft       - fourier-transformed 
        Ny          - number of datapoints
        filtu       - flag for filtering inputs; must be 0 for resting state
        filter_str  - filtering of signal threshold
        
    output:
        y_fft       - filtered fourier-transformed signal
        idx         - informative indices
    """
    # get dimensionality
    N, nr = y_fft.shape
    
    ## find freqs where h, u and y are present 
    # set precision
    prec = 1e-4

    # freq which are not smoothed out by convolution
    h_idx = np.where(np.abs(h_fft) > prec, True, False)
    
    # freq which are nonzero due to input
    if filtu == 1:
        u_idx = np.where(np.sum(np.abs(u_fft), axis = 1) > prec, True, False).reshape(N, 1)
        
    else:
        # just take all indices
        u_idx = np.ones((u_fft.shape[0],1), dtype = bool)
        

    # take all y indices
    y_idx = np.ones((u_fft.shape[0],1), dtype = bool)

    #indices where h_idx, u_idx, y_idx are present 
    str_idx = h_idx & u_idx & y_idx

    ## filtering by comparing to noise variance
    
    # relative threshold of signal compared to noise
    thr = filter_str
    
    # real and imaginary part
    y_real = y_fft.real
    y_imag = y_fft.imag
    
    # get standard deviation of noise signal
    if False in str_idx: # check if there are any 0s in str_idx
        std_real = np.std(y_real[~str_idx.flatten()], axis = 0, ddof = 1) #ddof needed for same std as in matlab
        std_imag = np.std(y_imag[~str_idx.flatten()], axis = 0, ddof = 1)
        
    else:
        std_real = np.zeros((N, nr))
        std_imag = np.zeros((N, nr))
        
    # find indices where real or imaginary signal is above threshold
    idx = np.where((np.abs(y_real) > thr*std_real) | (np.abs(y_imag) > thr*std_imag), True, False)
    
    ## high-pass filter       
    if filtu == 1:
        hpf = 16
        freq = 7*N//hpf
    
    else:
        hpf = np.maximum(16 + (thr-1)*4, 16)
        freq = 7*N//hpf        
    
    # high-pass filtering
    idx_freq = np.vstack((np.ones((1+freq, nr)), np.zeros((N - 2*freq - 1, nr)), np.ones((freq, nr))))
    idx_freq = idx_freq.astype(bool)

    idx = idx & np.tile(str_idx, nr) & idx_freq

    ## freqs that should be present due to fft constraints
    # constant frequency
    idx[0,:] = True

    # ensure symmetricity
    idx = idx | np.roll(idx[::-1], 1, axis = 0)

    # filter the data
    y_fft[~idx] = 0
    
    ## freq to include in regression (0s in informative regions are also kept)
    
    # iterate over regions
    for i in range(nr):
        
        # last freq in first half of spectrum
        first = np.max(np.nonzero(idx[:idx.shape[0]//2,i]))

        # first frequency in second half of the spectrum
        testy = np.nonzero(idx[idx.shape[0]//2:,i])
        idx.shape[0]//2
        #second = np.min(np.nonzero(idx[idx.shape[0]//2:,i])) + idx.shape[0]//2
        #print("region" + str(i) +": ", testy)
        second = np.min(np.array(testy)) + idx.shape[0]//2

    
        # set back to 1
        idx[:first, i] = 1
        idx[second:, i] = 1       
        
    return y_fft, idx
   

   
def reduce_zeros(X, Y):
    
    """
    Subsampling informative frequencies if there are more zero-valued frequencies
    to balance dataset. Basically some 0s get set to NaN

    input:
        X     - design matrix
        Y     - data
    
    output:
        Y     - balanced data
    """
    
    # combine X and Y
    XY = np.hstack((np.absolute(Y), np.absolute(X)))
    
    sum_xy = np.sum(XY, axis = 1)
             
    # get 0 indices
    idx_0 = np.where(sum_xy == 0)[0]
    
    # get number of 0 frequencies
    n0 = np.sum(sum_xy == 0)
    
    # get number of non-zero and non-nan frequencies
    n1 = np.sum(sum_xy > 0)
        
    # balance the data if too many zeros
    if n0 > n1:
        # draw n0-n1 samples from these indices
        idx_del = np.random.choice(idx_0, n0-n1, replace = False)
        # replace sampled indices with nan
        Y[idx_del] = np.nan

    return Y


if __name__ == "__main__":


    import scipy.io as sio
 
    """test cases"""
    """testing complete tutorial - generate + estimate"""
    
    # setup
    dcm = sio.loadmat('test_inputs/DCM_LargeScaleSmith_model1.mat')['DCM']    
    
    # structural priors on a, b, c, d
    a = dcm[0][0][0]
    b = dcm[0][0][1]
    c = dcm[0][0][2]
    d = dcm[0][0][3]
    
    # get U values
    U_u = np.asarray(dcm[0][0][4][0][0][1].todense())
    u_dt = 0.03125
    print(U_u.shape)
    
    # init U
    my_U = U(U_u, u_dt)

    # init DCM
    mydcm = DCM(a = a, b = b, c = c, d = d, U = my_U, nr = 50, v = 2714)    
    
    # get the Ep/Tp from 
    Tp_A = dcm[0][0][10][0][0][0]
    Tp_B = dcm[0][0][10][0][0][1]
    Tp_C = dcm[0][0][10][0][0][2]
    Tp_D = dcm[0][0][10][0][0][3]
    Tp_transit = dcm[0][0][10][0][0][4]
    Tp_decay = dcm[0][0][10][0][0][5]
    Tp_epsilon = dcm[0][0][10][0][0][6]
    
    Tp = Bunch(A = Tp_A,
                B = Tp_B,
                C = Tp_C,
                D = Tp_D,
                transit = Tp_transit,
                decay = Tp_decay,
                epsilon = Tp_epsilon)
    
    mydcm.Tp = Tp
    mydcm.set_options(type = 's', p0_all = 0.15, iter = 100, filter_str = 5, restrictInputs = 1)
    
    # run 
    y, y_noise, x = mydcm.generate(SNR = 3)    
    
    # estimate
    mydcm.estimate(set_options = False)
    
    # plotting, equivalent to mat-rDCM
    fig = plt.figure(figsize = (12,9))
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    # A-matrix, ground truth
    sns.heatmap(Tp_A, cmap = 'PiYG', vmax = 0.6, vmin = -0.6, ax = ax2)
    ax2.set(xlabel = "region (from)", ylabel  = "region (to)", title = "A-matrix, ground truth")
    
    # A-matrix, estimated
    sns.heatmap(mydcm.posterior.A, cmap = 'PiYG', vmax = 0.6, vmin = -0.6, ax = ax1)
    ax1.set(xlabel = "region (from)", ylabel  = "region (to)", title = "A-matrix, estimated")
    
    # true vs predicted time series from regions 0 and 11
    ax3.plot(np.append(y_noise[:,0], y_noise[:,11])  )      
    ax3.plot(np.append(mydcm.signal.y_pred_rdcm[:,0],mydcm.signal.y_pred_rdcm[:,11]))
    # add a vertical line to visually seperate the two regions
    ax3.vlines(len(y_noise[:,0]), -8, 6, color = "black")
    ax3.set(xlabel = "sample index", ylabel  = "BOLD", 
            xlim=(0,2*len(y_noise[:,0])), ylim = (-8, 6),
            title = "true and predicted time series")
    ax3.legend(["true", "predicted"])
    plt.show()


    # """Testing on real data"""
    
    ## real data, 1-n resting state
    from nilearn import datasets
    from nilearn.input_data import NiftiMapsMasker
    from nilearn.connectome import ConnectivityMeasure
    from nilearn import plotting

    print('Testing on real data')
    
    # get MSDL atlas from nilearn
    atlas = datasets.fetch_atlas_msdl()
    # Loading atlas image stored in 'maps'
    atlas_filename = atlas['maps']
    # Loading atlas data stored in 'labels'
    labels = atlas['labels']
    
    # path to preprocessed image
    img = "test_inputs/sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                          memory='nilearn_cache', verbose=5)
    
    # get region-wise BOLD
    time_series = masker.fit_transform(img)#, confounds=mean)
    
    # canonical fc
    print('calculating functional connectivity')
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    print('done!')
    # Mask out the major diagonal
    np.fill_diagonal(correlation_matrix, 0)
    # Display the correlation matrix
    plt.clf()
    plotting.plot_matrix(correlation_matrix, labels=labels, colorbar=True,
                      vmax=0.8, vmin=-0.8)
    plt.title("resting-state functional connectivity (Pearson correlation)")
    plt.show()
    
    #rDCM
    # initialize Y and U
    print('calculating directed connectivity')    
    my_Y = Y(y = time_series, dt = 3.56, names = labels) # Y from timeseries
    my_U = U(u = np.zeros((16*200, 25)), dt = 3.56/16) # empty U for resting-state

    # priors on ABCD, A is fully connected
    a = np.ones((39,39))
    b = np.zeros((39,39,25))
    c = np.zeros((39,25))
    d = np.zeros((39,39,39))
    
    # set up dcm
    mydcm = DCM(a = a, b = b, c = c, d = d, U = my_U, Y = my_Y, nr = 39)
    mydcm.nr = 39
    
    # options for resting state
    mydcm.set_options(type = 'r', filtu=0, filter_str = 5)
    # model inversion
    mydcm.estimate(set_options = False)
    print('done!')

    # mask A diagonal for visualization
    np.fill_diagonal(mydcm.posterior.A, 0)
    
    # plot A in nilearn style
    plt.clf()
    plotting.plot_matrix(mydcm.posterior.A, labels=labels, colorbar=True,
                      vmax = 0.05, vmin = -0.05)
    plt.title("resting-state directed connectivity (rDCM)")
