#---------------------------------------------------------------------------#
# TEST PARAMETERS WILL BE DELETED
# Xi_mat = np.array([[8.7255,-4.3753],[1,0]])
# Delta_mat = np.array([[4.3320,0],[0,1]])
# eVals, eVecs = la.eig(Xi_mat, Delta_mat)
# Xi_sortabs = sp.sort(abs(eVals))
# Xi_sortindex = sp.argsort(abs(eVals))
# Xi_sortedVec = sp.array([eVecs[:,i] for i in Xi_sortindex])
#---------------------------------------------------------------------------#


#-----------------------Uhlig's Example one parameters----------------------#
#N_bar     = 1.0/3
#Z_bar     = 1
#rho       = .36
#delta     = .025
#R_bar     = 1.01
#eta       =  1.0
#psi       = .95
#sigma_eps = .712
#
#betta   = 1.0/R_bar
#YK_bar  = (R_bar + delta - 1)/rho
#K_bar   = (YK_bar / Z_bar)**(1.0/(rho-1)) * N_bar
#I_bar   = delta * K_bar
#Y_bar   = YK_bar * K_bar
#C_bar   = Y_bar - delta*K_bar
#A       =  C_bar**(-eta) * (1 - rho) * Y_bar/N_bar
#
#
#
#
## for k_t
#AA = sp.array([[0],[-K_bar],[0],[0],[0]])
#
## for k_(t-1)
#BB = sp.array([[0],[(1-delta)*K_bar],[rho],[0],[-rho*YK_bar]])
#
#CC = sp.array([[-C_bar, Y_bar,0,0,-I_bar],[0, 0, 0, 0, I_bar],[0, -1, 1-rho, 0, 0,],[-eta, 1, -1, 0, 0],[0, rho*YK_bar,0,-R_bar,0]])
#
#DD = sp.array([[0],[0],[1],[0],[0]])
#
#FF = sp.array([0])
#GG = sp.array([0])
#HH = sp.array([0])
#
#JJ = sp.array([-eta,0,0,1,0])
#KK = sp.array([eta,0,0,0,0])
#LL = sp.array([0])
#MM = sp.array([0])
#NN = sp.array([psi])
#
#l_equ = sp.shape(AA)[0]
#m_states = sp.shape(AA)[1]
#
#l_equ = sp.shape(CC)[0]
#n_endog = sp.shape(CC)[1]
#
#l_equ = sp.shape(DD)[0]
#k_exog = sp.shape(CC)[1]
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
# l_equ = sp.shape(CC)[0]`
# n_endog = sp.shape(CC)[1]
#
# l_equ = sp.shape(DD)[0]
# k_exog = sp.shape(DD)[1]
#---------------------------------------------------------------------------#


#-----------------------Uhlig Example 6 Parameters--------------------------#
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