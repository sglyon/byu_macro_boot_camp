# -*- coding: utf-8 -*-

# IT WORKS!!!!!!!!!#
"""
Created on Mon May  7 16:06:36 2012

@author: Spencer Lyon
"""

import scipy as sp
from scipy import linalg as la
import numpy as np
from numpy import linalg as npla


def mparray2npfloat(a):
    """convert a NumPy array of mpmath objects (mpf/mpc)
    to a NumPy array of float"""
    tmp = a.ravel()
    res = _cfunc_float(tmp)
    return res.reshape(a.shape)

TOL = .000001
#---------------------------------------------------------------------------#
# TEST PARAMETERS WILL BE DELETED
# Xi_mat = np.array([[8.7255,-4.3753],[1,0]])
# Delta_mat = np.array([[4.3320,0],[0,1]])
# eVals, eVecs = la.eig(Xi_mat, Delta_mat)
# Xi_sortabs = sp.sort(abs(eVals))
# Xi_sortindex = sp.argsort(abs(eVals))
# Xi_sortedVec = sp.array([eVecs[:,i] for i in Xi_sortindex])
#---------------------------------------------------------------------------#


#---------------------------Uhlig's Example one parameters------------------#
# N_bar     = 1.0/3
# Z_bar     = 1
# rho       = .36
# delta     = .025
# R_bar     = 1.01
# eta       =  1.0
# psi       = .95
# sigma_eps = .712
# 
# betta   = 1.0/R_bar
# YK_bar  = (R_bar + delta - 1)/rho
# K_bar   = (YK_bar / Z_bar)**(1.0/(rho-1)) * N_bar
# I_bar   = delta * K_bar
# Y_bar   = YK_bar * K_bar
# C_bar   = Y_bar - delta*K_bar
# A       =  C_bar**(-eta) * (1 - rho) * Y_bar/N_bar
# 
# 
# 
# 
# # for k_t
# AA = sp.array([[0],[-K_bar],[0],[0],[0]])
# 
# # for k_(t-1)
# BB = sp.array([[0],[(1-delta)*K_bar],[rho],[0],[-rho*YK_bar]])
# 
# CC = sp.array([[-C_bar, Y_bar,0,0,-I_bar],[0, 0, 0, 0, I_bar],[0, -1, 1-rho, 0, 0,],[-eta, 1, -1, 0, 0],[0, rho*YK_bar,0,-R_bar,0]])
# 
# DD = sp.array([[0],[0],[1],[0],[0]])
# 
# FF = sp.array([0])
# GG = sp.array([0])
# HH = sp.array([0])
# 
# JJ = sp.array([-eta,0,0,1,0])
# KK = sp.array([eta,0,0,0,0])
# LL = sp.array([0])
# MM = sp.array([0])
# NN = sp.array([psi])
# 
# l_equ = sp.shape(AA)[0]
# m_states = sp.shape(AA)[1]
# 
# l_equ = sp.shape(CC)[0]
# n_endog = sp.shape(CC)[1]
# 
# l_equ = sp.shape(DD)[0]
# k_exog = sp.shape(CC)[1]
#---------------------------------------------------------------------------#

#-----------------------Uhlig Example 5 Parameters--------------------------#
# N_bar     = 1.0/3
# Z_bar     = 1
# rho       = .36
# R_bar     = 1.01
# eta       = 1.0
# psi       = .95
# sigma_eps = .712
# p_echo    = 4
# betta   = 1.0/R_bar
# YK_bar  = (1- betta)/((1 - betta**p_echo)*betta*rho)
# K_bar   = (YK_bar / Z_bar)**(1.0/(rho-1)) * N_bar
# I_bar   = K_bar / p_echo 
# Y_bar   = YK_bar * K_bar
# C_bar   = Y_bar - I_bar
# Lam_bar = C_bar**(- eta)
# Mu_bar  = rho*Lam_bar*YK_bar
# A       = Lam_bar * (1 - rho) * Y_bar/N_bar
# 
# AA = sp.array([[-I_bar, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, betta**2, betta**3, betta**4],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, ],[0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0, ]])
# BB = sp.array([[0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[-1, 0, 0, 0, 0, 0, 0, ],[0, -1, 0, 0, 0, 0, 0],[0, 0, -1, 0, 0, 0, 0, ]])
# CC = sp.array([[-C_bar, Y_bar, 0, 0, 0, 0, 0],[0, 0, -p_echo, 0, 0, 0, 0],[0, -1, rho,(1-rho), 0, 0, 0],[0, 1, 0, -1, 1, 0, 0],[0, 0, 0,0,(-Lam_bar/Mu_bar), 0, betta],[0, 1, -1, 0, 1, -1, 0],[eta, 0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0]])
# DD = sp.array([[0],[0],[1],[0],[0],[0],[0],[0],[0],[0]])
# FF = sp.array([[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,-1,0,0],[0,0,0,0,0,-1,0]])
# GG = sp.array([[0,0,0,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
# HH = sp.zeros((4,7))
# JJ = sp.array([[0, 0, 0, 0, 0, -1, 0],[0, 0, 0, 0, 0, 0, -1],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0]])
# KK = sp.array([[0, 0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0]])
# LL = sp.array([[0],[0],[0],[0]])
# MM = sp.array([[0],[0],[0],[0]])
# NN = sp.mat([psi])
# Sigma = sp.array([sigma_eps**2])
# 
# l_equ = sp.shape(AA)[0]
# m_states = sp.shape(AA)[1]
# 
# l_equ = sp.shape(CC)[0]
# n_endog = sp.shape(CC)[1]
# 
# l_equ = sp.shape(DD)[0]
# k_exog = sp.shape(DD)[1]
#---------------------------------------------------------------------------#


#-------------------------# Uhlig Example 6---------------------------------#
# Z_bar = 1
# NPV_frac = 0.5
# rho = 0.36
# delta = 0.025
# R_bar = 1.01
# eta = 1.0
# theta = 0.8
# psi_z = 0.95
# sigma_z   = .712
# psi_r     = .95
# sigma_r   = 1.0
# corr_z_r  = 0
# 
# betta   = 1.0/R_bar
# XK_bar  = ((1+delta)**theta - 1)**(1.0/theta)
# FK_omt  = (1+delta)**(1-theta)
# FX_omt  = FK_omt/XK_bar**(1-theta)
# YK_bar  = (R_bar - FK_omt + delta)/(rho*FX_omt)
# K_bar   = (Z_bar/YK_bar)**(1.0/(1-rho))
# Y_bar   = Z_bar*K_bar**rho
# X_bar   = XK_bar*K_bar
# F_bar   = (1+delta)*K_bar
# A_bar   = NPV_frac * (Y_bar/(R_bar - 1))
# C_bar   = Y_bar - X_bar + (R_bar - 1)*A_bar
# 
# AA = sp.array([[0,-A_bar],[0,0],[-1,0],[0,0]])
# BB = sp.array([[0, R_bar*A_bar],[rho, 0],[-delta, 0],[K_bar**theta, 0]])
# CC = sp.array([[-C_bar, -X_bar, Y_bar, 0],[0, 0, -1, 0],[0, 0, 0, (1+delta)],[0, X_bar**theta, 0, (-F_bar**theta)]])
# DD = sp.array([[0, R_bar*A_bar],[1, 0],[0, 0],[0, 0]])
# FF = sp.array([[0, 0],[0, 0]])
# GG = sp.array([[0, 0],[( -((rho/R_bar)*YK_bar*FX_omt + (1-theta)*FK_omt/R_bar) ), 0]])
# HH = sp.array([[0, 0],[0, 0]])
# JJ = sp.array([[-eta, 0, 0, 0],[-eta,(1-theta)*(FK_omt-delta)/R_bar,(rho/R_bar)*YK_bar*FX_omt,delta*(1-theta)/R_bar]])
# KK = sp.array([[eta, 0, 0, 0],[eta, theta-1, 0, 1-theta]])
# LL = sp.array([[0, 1],[0, 0]])
# MM = sp.array([[0, 0],[0, 0]])
# NN = sp.array([[psi_z,0],[0, psi_r]])
# Simga = ([[sigma_z**2, corr_z_r*sigma_z*sigma_r],[corr_z_r*sigma_z*sigma_r, sigma_r**2]])
# 
# l_equ = sp.shape(AA)[0]
# m_states = sp.shape(AA)[1]
# 
# l_equ = sp.shape(CC)[0]
# n_endog = sp.shape(CC)[1]
# 
# l_equ = sp.shape(DD)[0]
# k_exog = sp.shape(DD)[1]
#---------------------------------------------------------------------------#



def nullSpaceBasis(A):
    U,s, Vh = la.svd(A)
    vecs = np.array([])
    toAppend = A.shape[1] -s.size
    s = sp.append(s,sp.zeros((1,toAppend)))
    counter = 0
    for i in range(0,s.size):
        if s[i]==0:
            vecs = Vh[-toAppend:,:]
    if vecs.size ==0:
        vecs = sp.zeros((1,A.shape[1]))
    return sp.mat(vecs)
	
def solve_P(F,G,H):
    """This function takes arguments for F,G,H and solves the matrix quadratic given by
    F*P^2+G*P+H=0.  Note F, G, and H must be square.
    The function returns the matrix P and the resulting matrix, given by F*P^2+G*P+H 
    which should be close to zero.
    The algorithm used to solve for P is outlined in 'A Toolkit for Analyzing Nonlinear 
    Dynamic Stochastic Models Easily' by Harald Uhlig.
    """
    m=sp.shape(F)[0]
    
    Xi=sp.concatenate((-G,-H), axis=1)
    second=sp.concatenate((sp.eye(m,m),sp.zeros((m,m))),axis=1)
    Xi=sp.concatenate((Xi,second))
    
    Delta=sp.concatenate((F,sp.zeros((m,m))),axis=1)
    second=sp.concatenate((sp.zeros((m,m)),sp.eye(m,m)),axis=1)
    Delta=sp.concatenate((Delta,second))
    
    (L,V) = la.eig(Xi,Delta)
    
    boolean = sp.zeros(len(L))
    trueCount =0
    
    for i in range(len(L)):
        if L[i]<1 and L[i]>-1 and sp.imag(L[i])==0 and trueCount<m:
            boolean[i] = True
            trueCount+=1
    #display(L, boolean)
    if trueCount<m:
        print "Imaginary eigenvalues being used"
        for i in range(len(L)):
            if math.sqrt(real(L[i])**2+imag(L[i])**2)<1 and trueCount<m:
                boolean[i]=True
                trueCount+=1
    #display(boolean)
    
    if trueCount==m:
        print "true count is m"
        Omega=sp.zeros((m,m))
        diagonal=[]
        count =0
        for i in range(len(L)):
            if boolean[i]==1:
                Omega[:,count]=sp.real(V[m:2*m,i])+sp.imag(V[m:2*m,i])
                diagonal.append(L[i])
                count+=1
        Lambda=sp.diag(diagonal)
        try:
            P=sp.dot(sp.dot(Omega,Lambda),la.inv(Omega))
        except:
            print 'Omega not invertable'
            P=sp.zeros((m,m))
        diff=sp.dot(F,sp.dot(P,P))+sp.dot(G,P)+H
        return P,diff
    else:
        print "Problem with input, not enough 'good' eigenvalues"
        return sp.zeros((m,m)),sp.ones((m,m))*100
		
def solvingStuffForPQRS(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, NN):
    q_eqns = sp.shape(FF)[0]
    m_states = sp.shape(FF)[1]
    l_equ = sp.shape(CC)[0]
    n_endog = sp.shape(CC)[1]

	
    k_exog = min(sp.shape(sp.mat(NN))[0], sp.shape(sp.mat(NN))[1])


    if n_endog > 0 and npla.matrix_rank(CC)<n_endog:
        print('CC is not of rank n. Cannot solve. Quitting. Goodbye')
    else:
        if l_equ==0:
#            CC_plus = la.pinv(CC)
 #           CC_0 = nullSpaceBasis(CC.T)
            Psi_mat = FF
            Gamma_mat=- GG
            Theta_mat = -HH
            Xi_mat    = sp.vstack(( sp.hstack((Gamma_mat, Theta_mat)), sp.hstack((sp.eye(m_states), sp.zeros((m_states,m_states))))))
            Delta_mat = sp.vstack((sp.hstack((Psi_mat, sp.mat(sp.zeros((m_states,m_states))))), sp.hstack((sp.zeros((m_states,m_states)), sp.eye(m_states))))) 
        
        else:
            CC_plus = la.pinv(CC)
            CC_0 = nullSpaceBasis(CC.T)
            Psi_mat = sp.vstack((sp.zeros((l_equ - n_endog, m_states)), FF - sp.dot(sp.dot(JJ,CC_plus),AA)))
            if sp.size(CC_0) !=0:
                Gamma_mat = sp.vstack((sp.dot(CC_0,AA), sp.dot(sp.dot(JJ,CC_plus),BB) - GG + sp.dot(sp.dot(KK,CC_plus),AA) ))
                Theta_mat = sp.vstack((sp.dot(CC_0,AA), sp.dot(sp.dot(KK,CC_plus),BB) - HH))
            else:
                Gamma_mat = sp.dot(sp.dot(JJ,CC_plus),BB) - GG + sp.dot(sp.dot(KK,CC_plus),AA)
                Theta_mat = sp.dot(sp.dot(KK,CC_plus),BB) - HH
            Xi_mat    = sp.vstack(( sp.hstack((Gamma_mat, Theta_mat)), sp.hstack((sp.eye(m_states), sp.zeros((m_states,m_states))))))
            Delta_mat = sp.vstack((sp.hstack((Psi_mat, sp.mat(sp.zeros((m_states,m_states))))), sp.hstack((sp.zeros((m_states,m_states)), sp.eye(m_states)))))
        
        eVals, eVecs = la.eig(Xi_mat, Delta_mat)
        if npla.matrix_rank(eVecs)<m_states:
            print('Error: Xi is not diagonalizable, stopping')
        else:
            Xi_sortabs = sp.sort(abs(eVals))
            Xi_sortindex = sp.argsort(abs(eVals))
            Xi_sortedVec = sp.array([eVecs[:,i] for i in Xi_sortindex])
            Xi_sortval = Xi_sortabs
            Xi_select = np.arange(0,m_states)
            if sp.imag(Xi_sortedVec[m_states-1]).any():
                if (abs(Xi_sortval[m_states-1] - sp.conj(Xi_sortval[m_states])) <TOL):
                    drop_index = 1
                    while (abs(sp.imag(Xi_sortval[drop_index]))>TOL) and (drop_index < m_states):
                        drop_index+=1
                    if drop_index >= m_states:
                        print('There is an error. Too many complex eigenvalues. Quitting')
                    else:
                        print('droping the lowest real eigenvalue. Beware of sunspots')
                        Xi_select = np.array([np.arange(0,drop_index-1),np.arange(drop_index+1,m_states)])
            # Here Uhlig computes stuff if user chose "Manual roots" I am skipping it.
            if max(abs(Xi_sortval[Xi_select]))> 1 + TOL:
                print('It looks like we have unstable roots. This might not work')
            if abs(max(abs(Xi_sortval[Xi_select])) -1) < TOL:
                print('Check the model to make sure you have a unique steady state'\
                ,'we are having problems with convergence.')
            Lambda_mat = sp.diag(Xi_sortval[Xi_select])
            Omega_mat = sp.mat(sp.hstack((Xi_sortedVec[(m_states):(2*(m_states)),Xi_select])))
            print Omega_mat
            # Omega_mat = sp.reshape(Omega_mat,(Omega_mat.size**1/2,Omega_mat.size**1/2))
            if npla.matrix_rank(Omega_mat) < m_states:
                PP, Diff = solve_P(FF, GG, HH)
                PP = sp.real(PP)
                FFPP_plus_GG = sp.dot(FF,PP) + GG
                print FFPP_plus_GG
                print PP
                print("Omega matrix is not invertible, Can't solve for P")
            else:
                PP =  sp.dot(sp.dot(Omega_mat,Lambda_mat),la.inv(Omega_mat))
                PP_imag = sp.imag(PP)
                PP = sp.real(PP)
                if sum(sum(abs(PP_imag))) / sum(sum(abs(PP))) > .000001:
                    print("A lot of P is complex. We will continue with the real part and hope we don't lose too much information")
                    
        #The code below was from he Uhlig file cacl_qrs.m. I think for python it fits better here.
    if l_equ ==0:
        RR = sp.zeros((0,m_states))
        Bleh = (la.kron(sp.mat(NN.T), sp.mat(FF)))
        print Bleh
        #Don't you dare delete bleh and meh... They are the reason it works!!!
        Meh = la.kron(NN.T, FFPP_plus_GG)
		#100000000 strong for the matrices bleh and meh.  Click HERE to join
        print FFPP_plus_GG
        VV = Bleh + Meh
#        VV = mparray2npfloat(VV)
        print VV, 'death'
		#VV = sp.kron(NN.T,FF)+sp.kron(sp.eye(k_exog),sp.dot(FF,PP)+GG) #sp.kron(NN.T,JJ)+sp.kron(sp.eye(k_exog),KK)))
    else:   
        RR = - sp.dot(CC_plus,(sp.dot(AA,PP)+BB))
        VV = sp.vstack((sp.hstack((sp.kron(eye(k_exog),AA), sp.kron(eye(k_exog),CC))),\
        sp.hstack((sp.kron(NN.T,FF)+sp.kron(sp.eye(k_exog),sp.dot(FF,PP)+sp.dot(JJ,RR)+GG), sp.kron(NN.T,JJ)+sp.kron(sp.eye(k_exog),KK)))))
        
    if  False and (npla.matrix_rank(VV) < k_exog*(m_states + n_endog)):
        print("Sorry but V is note invertible. Can't solve for Q and S")
    else:
        LLNN_plus_MM = sp.dot(sp.mat(LL.T),sp.mat(NN)) + sp.mat(MM.T)
        QQSS_vec = sp.dot(la.inv(sp.mat(VV)), sp.mat(LLNN_plus_MM))
        QQSS_vec = -QQSS_vec
        if max(abs(QQSS_vec)) == sp.inf:
            print("We have issues with Q and S. Entries are undefined. Probably because V is no inverible.")
        
        QQ = sp.reshape(QQSS_vec[0:m_states*k_exog],(m_states,k_exog))
        SS = sp.reshape(QQSS_vec[(m_states*k_exog):((m_states+n_endog)*k_exog)],(n_endog,k_exog))
        
        # The vstack and hstack's below are ugly, but safe. If you have issues with WW uncomment the first definition and comment out the second one.
        #WW = sp.vstack((\
        #sp.hstack((sp.eye(m_states), sp.zeros((m_states,k_exog)))),\
        #sp.hstack((sp.dot(RR,sp.pinv(PP)), (SS-sp.dot(sp.dot(RR,sp.pinv(PP)),QQ)))),\
        #sp.hstack((sp.zeros((k_exog,m_states)),sp.eye(k_exog)))))
        
        WW = sp.array([[sp.eye(m_states), sp.zeros((m_states,k_exog))],\
        [sp.dot(RR,la.pinv(PP)), (SS-sp.dot(sp.dot(RR,la.pinv(PP)),QQ))],\
        [sp.zeros((k_exog,m_states)),sp.eye(k_exog)]])
    return PP, QQ, RR, SS
    
#def solveQRS():
    #if l_equ ==0:
    #    RR = sp.zeros((0,m_states))
    #    VV = sp.hstack((sp.kron(NN.T,FF)+sp.kron(sp.eye(k_exog),sp.dot(FF,PP)+GG), sp.kron(NN.T,JJ)+sp.kron(sp.eye(k_exog),KK)))
    #else:   
    #    RR = - sp.dot(CC_plus,(sp.dot(AA,PP)+BB))
    #    VV = sp.vstack((sp.hstack((sp.kron(eye(k_exog),AA), sp.kron(eye(k_exog),CC))),\
    #    sp.hstack((sp.kron(NN.T,FF)+sp.kron(sp.eye(k_exog),sp.dot(FF,PP)+sp.dot(JJ,RR)+GG), sp.kron(NN.T,JJ)+sp.kron(sp.eye(k_exog),KK)))))
    #    
    #if (np.matrix_rank(VV) < k_exog*(m_states + n_endog)):
    #    print("Sorry but V is note invertible. Can't solve for Q and S")
    #else:
    #    LLNN_plus_MM = LL*NN + MM
    #    QQSS_vec = - sp.dot(la.inv(VV), sp.vstack((DD,LLNN_plus_MM )))
    #    if max(abs(QQSS_vec)) == sp.inf:
    #        print("We have issueswith Q and S. Entries are undefined. Probably because V is no inverible.")
    #    
    #    QQ = sp.reshape(QQSS_vec[0:m_states*k_exog],(m_states,k_exog))
    #    SS = sp.reshape(QQSS_vec[(m_states*k_exog+1):((m_states+n_endog)*k_endog)],(n_endog,k_endog))
    #    WW = sp.vstack((\
    #    sp.hstack((sp.eye(m_states), sp.zeros((m_states,k_exog)))),\
    #    sp.hstack((sp.dot(RR,sp.pinv(PP)), (SS-sp.dot(sp.dot(RR,sp.pinv(PP)),QQ)))),\
    #    sp.hstack((sp.zeros((k_exog,m_states)),sp.eye(k_exog)))))
    #    
        
    