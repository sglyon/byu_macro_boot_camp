# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:06:36 2012

@author: Spencer Lyon
"""

import scipy as sp
from scipy import linalg as la
import numpy as np
from numpy import linalg as npla

def nullSpaceBasis(A):
    """
    This funciton will find the basis of the null space of the matrix A.

    Inputs:
        A: The matrix you want the basis for
    Outputs:
        A numpy matrix containing the vectors as row vectors.

    Notes:
        If A is an empty matrix, an empty matrix is returned.

    """
    if A:
        U,s, Vh = la.svd(A)
        vecs = np.array([])
        toAppend = A.shape[1] -s.size
        s = sp.append(s,sp.zeros((1,toAppend)))
        for i in range(0,s.size):
            if s[i]==0:
                vecs = Vh[-toAppend:,:]
        if vecs.size ==0:
            vecs = sp.zeros((1,A.shape[1]))
        return sp.mat(vecs)
    else:
        return sp.zeros((0,0))

def solvingStuffForPQRS(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, NN):
    """
    This function mimics the behavior of Harald Uhlig's solve.m and calc_qrs.m
    files in Uhlig's toolkit.

    In order to use this function, the user must have log-linearized the model
    they are dealing with to be in the following form (assume that y corresponds
    to the model's "jump variables", z represents the exogenous state
    variables and x is for endogenous state variables. nx, ny, nz correspond
    to the number of variables in each category.)
        A*x_t + B*x_t-1 + C* y_t + D*z_t = 0

        E{F*x_t+1 + G*x_t + H*x_t-1 + J*y_t+1 + K*y_t + L*z_t+1 K*z_t} = 0

    Inputs:
        AA: The ny x nx numpy matrix represented above by A
        BB: The ny x nx numpy matrix represented above by B
        CC: The ny x ny numpy matrix represented above by C
        DD: The ny x nz numpy matrix represented above by D

        FF: The nx x nx numpy matrix represented above by F
        GG: The nx x nx numpy matrix represented above by G
        HH: The nx x nx numpy matrix represented above by H
        JJ: The nx x ny numpy matrix represented above by J
        KK: The nx x ny numpy matrix represented above by K
        LL: The nx x nz numpy matrix represented above by L
        MM: The nx x nz numpy matrix represented above by M
        NN: The autocorrelation numpy matrix for the exogenous state vector z.

    Outputs:
        The purpose of this routine is to find the matrices P and Q such that
        the following matrix quadratic equations are satisfied:

            F*P**2 + G*g + H =0
            F*Q*N + (F*P+G)*Q + (L*N + M) =0

        P: The P matrix that satisfies the above equations
        Q: The Q matrix that satisfies the above equations.

    Notes:
        Make sure that AA-NN are all numpy matrices, especially if some of them
        are empty. We often need dimensions and shapes and a numpy matrix
        ensures we can get the data we need without errors that would result
        if numpy arrays were to be passed in.


    """
    TOL = .000001

    # Here we use matrices to get pertinent dimensions.
    nx = sp.mat(FF).shape[1]
    l_equ = sp.shape(CC)[0]
    ny = sp.shape(CC)[1]

    k_exog = min(sp.shape(sp.mat(NN))[0], sp.shape(sp.mat(NN))[1])

    # The following if and else blocks form the Psi, Gamma, Theta Xi, Delta mats
    if l_equ==0:
        if CC.any():
            # This blcok makes sure you don't throw an error with an empty CC.
            CC_plus = la.pinv(CC)
            CC_0 = nullSpaceBasis(CC.T)
        else:
            CC_plus = sp.mat([])
            CC_plus = sp.mat([])
        Psi_mat = FF
        Gamma_mat=- GG
        Theta_mat = -HH
        Xi_mat    = sp.mat(sp.vstack(( sp.hstack((Gamma_mat, Theta_mat)), sp.hstack((sp.eye(nx), sp.zeros((nx,nx)))))))
        Delta_mat = sp.mat(sp.vstack((sp.hstack((Psi_mat, sp.mat(sp.zeros((nx,nx))))), sp.hstack((sp.zeros((nx,nx)), sp.eye(nx))))))

    else:
        CC_plus = la.pinv(CC)
        CC_0 = nullSpaceBasis(CC.T)
        Psi_mat = sp.vstack((sp.zeros((l_equ - ny, nx)), FF - sp.dot(sp.dot(JJ,CC_plus),AA)))
        if sp.size(CC_0) ==0:
            # This block makes sure you don't throw an error with an empty CC.
            Gamma_mat = sp.vstack((sp.dot(CC_0,AA), sp.dot(sp.dot(JJ,CC_plus),BB) - GG + sp.dot(sp.dot(KK,CC_plus),AA) ))
            Theta_mat = sp.vstack((sp.dot(CC_0,AA), sp.dot(sp.dot(KK,CC_plus),BB) - HH))
        else:
            Gamma_mat = sp.dot(sp.dot(JJ,CC_plus),BB) - GG + sp.dot(sp.dot(KK,CC_plus),AA)
            Theta_mat = sp.dot(sp.dot(KK,CC_plus),BB) - HH
        Xi_mat    = sp.vstack(( sp.hstack((Gamma_mat, Theta_mat)), sp.hstack((sp.eye(nx), sp.zeros((nx,nx))))))
        Delta_mat = sp.vstack((sp.hstack((Psi_mat, sp.mat(sp.zeros((nx,nx))))), sp.hstack((sp.zeros((nx,nx)), sp.eye(nx)))))

    # Now we need the generalized eigenvalues/vectors for Xi with respect to
    # Delta. That is eVals and eVecs below.

    eVals, eVecs = la.eig(Xi_mat, Delta_mat)
    if npla.matrix_rank(eVecs)<nx:
        print('Error: Xi is not diagonalizable, stopping')

    # From here to line 158 we Diagonalize Xi, form Lambda/Omega and find P.
    else:
        Xi_sortabs = sp.sort(abs(eVals))
        Xi_sortindex = sp.argsort(abs(eVals))
        Xi_sortedVec = sp.array([eVecs[:,i] for i in Xi_sortindex]).T
        Xi_sortval = Xi_sortabs
        Xi_select = np.arange(0,nx)
        if sp.imag(Xi_sortedVec[nx-1]).any():
            if (abs(Xi_sortval[nx-1] - sp.conj(Xi_sortval[nx])) <TOL):
                drop_index = 1
                while (abs(sp.imag(Xi_sortval[drop_index]))>TOL) and (drop_index < nx):
                    drop_index+=1
                if drop_index >= nx:
                    print('There is an error. Too many complex eigenvalues. Quitting')
                else:
                    print('droping the lowest real eigenvalue. Beware of sunspots')
                    Xi_select = np.array([np.arange(0,drop_index-1),np.arange(drop_index+1,nx)])
        # Here Uhlig computes stuff if user chose "Manual roots" I am skipping it.
        if max(abs(Xi_sortval[Xi_select]))> 1 + TOL:
            print('It looks like we have unstable roots. This might not work')
        if abs(max(abs(Xi_sortval[Xi_select])) -1) < TOL:
            print('Check the model to make sure you have a unique steady state'\
            ,'we are having problems with convergence.')
        Lambda_mat = sp.diag(Xi_sortval[Xi_select])
        Omega_mat = sp.mat(sp.hstack((Xi_sortedVec[(nx):(2*(nx)),Xi_select])))
        Omega_mat = sp.reshape(Omega_mat,(Omega_mat.size**1/2,Omega_mat.size**1/2))
        if npla.matrix_rank(Omega_mat) < nx:
            print("Omega matrix is not invertible, Can't solve for P")
        else:
            PP =  sp.dot(sp.dot(Omega_mat,Lambda_mat),la.inv(Omega_mat))
            PP_imag = sp.imag(PP)
            PP = sp.real(PP)
            if (sum(sum(abs(PP_imag))) / sum(sum(abs(PP))) > .000001).any():
                print("A lot of P is complex. We will continue with the real part and hope we don't lose too much information")

    # The code from here to the end was from he Uhlig file cacl_qrs.m.
    # I think for python it fits better here than in a separate file.

    # The if and else below make RR and VV depending on our model's setup.
    if l_equ ==0:
        RR = sp.zeros((0,nx))
        VV = sp.hstack((sp.kron(NN.T,FF)+sp.kron(sp.eye(k_exog),sp.dot(FF,PP)+GG), sp.kron(NN.T,JJ)+sp.kron(sp.eye(k_exog),KK)))
    else:
        RR = - sp.dot(CC_plus,(sp.dot(AA,PP)+BB))
        VV = sp.vstack((sp.hstack((sp.kron(sp.eye(k_exog),AA), sp.kron(sp.eye(k_exog),CC))),\
        sp.hstack((sp.kron(NN.T,FF)+sp.kron(sp.eye(k_exog),sp.dot(FF,PP)+sp.dot(JJ,RR)+GG), sp.kron(NN.T,JJ)+sp.kron(sp.eye(k_exog),KK)))))

    # Now we use LL, NN, RR, VV to get the QQ, RR, SS matrices.
    if (npla.matrix_rank(VV) < k_exog*(nx + ny)):
        print("Sorry but V is note invertible. Can't solve for Q and S")
    else:
        LL = sp.mat(LL)
        NN = sp.mat(NN)
        LLNN_plus_MM = sp.dot(LL,NN) + MM
        QQSS_vec = - sp.dot(la.inv(VV), sp.vstack((DD,LLNN_plus_MM )))
        if (max(abs(QQSS_vec)) == sp.inf).any():
            print("We have issues with Q and S. Entries are undefined. Probably because V is no inverible.")

        QQ = sp.reshape(QQSS_vec[0:nx*k_exog],(nx,k_exog))
        SS = sp.reshape(QQSS_vec[(nx*k_exog):((nx+ny)*k_exog)],(ny,k_exog))

        # Not sure what WW is for? Uhlig had it so I copied it in case we needed
        # it in the future.
        WW = sp.vstack((\
        sp.hstack((sp.eye(nx), sp.zeros((nx,k_exog)))),\
        sp.hstack((sp.dot(RR,la.pinv(PP)), (SS-sp.dot(sp.dot(RR,la.pinv(PP)),QQ)))),\
        sp.hstack((sp.zeros((k_exog,nx)),sp.eye(k_exog)))))

    return PP, QQ, RR, SS
