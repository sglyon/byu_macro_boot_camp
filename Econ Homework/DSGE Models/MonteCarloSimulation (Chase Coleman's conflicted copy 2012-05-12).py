import scipy as sp
import math
from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from scipy import mean

def Xgen(X0, Z, PP, QQ, Xbar):
    """
    This function generates a history of X given a history
    technology shocks (Z), a P matrix, a Q matrix, and an
    intial X (X0).
    Note Xt(tilde) = PXt-1(tilde) +QZt(tilde)
    Xt=Xbar*e^Xt(tilde)
    """
    num_endog=sp.shape(PP)[1]
    T=len(Z) 
    X=sp.zeros((num_endog,T))
    X[:,0]=X0
    for i in range(1,T):
        Zt=Z[i]
        Xt_1=sp.zeros((num_endog,1))
        for j in range(num_endog):
            Xt_1[j,0]=X[j,i-1]
        Xt=sp.dot(PP,Xt_1)+sp.dot(QQ,Zt)
        for k in range(num_endog):
            X[k,i]=Xt[k,0]
    exponents=sp.exp(X)
    for p in range(T):
        for q in range(num_endog):
            X[q,p]=Xbar[0,q]*exponents[q,p]
    return X

def Zgen(Initial, RHOz, MUz, VARz, zbar, t):
	"""
	Zt = RHOz*Ztm1 + EPSz
	This randomly generates the technology shocks given
	a specified RHOz (=N) and EPSz i.i.d.~(MUz,VARz) for
	a specified number of periods t.
	Z0 = 0
	Z1 = Initial
	"""
	#Standard deviation of Z needed for future application (gauss)
	SIGz = math.sqrt(VARz)
	
	#Generate Z for t periods, we will populate each period below
	Z = sp.zeros(t)
	
	#The first period of Z we assume all technology shock is 0
	#The second period the Z = indicated initial value
	
	Z[0] = 0
	Z[1] = Initial
	
	
	#Zt = RHOz*Ztm1 + EPSz where EPSz i.i.d.~(MUz, VARz)
	for i in range(2,t):
		Z[i] = RHOz*Z[i-1] + gauss(MUz,SIGz)
    for j in range(t):
        Z[j]=Z[j]+zbar
	#Should give you what you want! :)
	return Z

def Ygen(Y0,Z,RR,SS):
    """
    This function generates a history of Y given a history
    technology shocks (Z), a R matrix, a S matrix, and an
    intial Y (Y0).
    Note Yt = RXt-1 +SZt
	
	TODO: Check to see if we need to raise y to exp like x above?
    """
    num_exog=sp.shape(RR)[1]
    T=sp.shape(Z)[1]
    Y=sp.zeros((num_exog,T))
    Y[:,0]=Y0
    for i in range(1,T):
        Zt=Z[0,i]
        Yt_1=sp.zeros((num_exog,1))
        for j in range(num_exog):
            Yt_1[j,0]=Y[j,i-1]
        Yt=sp.dot(RR,Yt_1)+sp.dot(SS,Zt)
        for k in range(num_exog):
            Y[k,i]=Yt[k,0]
    return Y


def MonteCarlo(P,Q,R,S,N,X0,Y0,Z0,Xbar,Zbar,Mu,Var,alpha,delta,reps,T,full=2):
    """
    Runs Monte Carlo simulation for DSGE model.
    Inputs:
        P,Q,R,S,N : Matrices previously found from steady state and solving
                    the matrix quadratic equations.
        X0,Y0,Z0 : Intial values for X,Y,Z histories
        Xbar, Zbar: Steady State Values for X and Z
        Mu : Mean value of epsilon shocks
        Var : Variance of epsilon shocks
        alpha, delta : constants from the model
        reps : Number of simulations to run
        T : time periods to run the simulation for
        full : integer, if changed to 1, the model is run for the specified time periods
            if left at 2, the model is run for twice the specified time periods and graphs
            are shown for the second half of the time periods
    Outputs:
        GDP,I,C: GDP, investment, and consumption matrices, each row is a time period,
                columns are simulations
        Displays the following graphs:
            GDP, Investment, Consumption with Means, 5% and 95% confidence bands
        Prints mean and standard deviations of the following moments:
            mean, standard deviation, correlations between GDP and investment and consumption,
            and autocorrelations for GDP, investment and consumption
        Creates the following CSV files:
            GDPAUTO.csv (autocorrelation of GDP)
            IAUTO.csv (autocorrelation of investment)
            CAUTO.csv (autocorrelation of consumption)
            GDPICORR.csv (correlation of GDP and investment)
            GDPCCORR.csv (correlation of GDP and consumption)

    Notes:
        Make sure that from whatever file you are calling this from you have
        created a function called YICgen and declared it as global just above
        its definition by using the line of code : 'global YICgen'
    """
    T=full*T

    GDP = sp.zeros((T,reps))
    I = sp.zeros((T,reps))
    C = sp.zeros((T,reps))

    GDPIcorr = []
    GDPCcorr = []

    GDPauto = []
    Iauto = []
    Cauto = []

    GDPmean = []
    Imean = []
    Cmean = []

    GDPstd = []
    Istd = []
    Cstd = []


    for i in range(reps):
        #generate X,Y,Z
        Z=[]
        X=[]
        Z = Zgen(Z0,N,Mu,Var,Zbar, T)
        X = Xgen(X0,Z,P,Q,Xbar)

        #find y,i,c for each time period
        #GDP[:,i],invest[:,i],consumption[:,i] = YICgen(X,Z,alpha,delta)
        gdp,invest,consumption=YICgen(X,Z,alpha,delta)

        #Correlation between output and investment for each series
        gdpi = sp.corrcoef(gdp,invest)
        GDPIcorr.append(gdpi[0,1])
	#Correlation between output and consumption for each series
	gdpc = sp.corrcoef(gdp,consumption)
	GDPCcorr.append(gdpc[0,1])

        #GDP autocorrelation coefficients for each series appended
	#to the empty GDPauto list
        gauto = sp.corrcoef(gdp[0:-1],gdp[1:])
	GDPauto.append(gauto[0,1])

        #Investment autocorrelation coefficients for each series
	#appended to the empty Iauto list
	iauto = sp.corrcoef(invest[0:-1],invest[1:])
	Iauto.append(iauto[0,1])

        #Consumption autocorrelation coefficients for each series
	#appended to the empty Cauto list
	cauto = sp.corrcoef(consumption[0:-1],consumption[1:])
	Cauto.append(cauto[0,1])

        #means across T for each simulation
        GDPmean.append(sp.mean(gdp))
        Imean.append(sp.mean(invest))
        Cmean.append(sp.mean(consumption))

        #standard deviations across T for each simulation
        GDPstd.append(sp.std(gdp))
        Istd.append(sp.std(invest))
        Cstd.append(sp.std(consumption))

        #for total list at the end
        GDP[:,i]=gdp
        I[:,i]=invest
        C[:,i]=consumption

        #Y = Ygen(Y0,Z,R,S)

    #Mean and standard deviation of correlation between GDP and
    #Investment and Consumption over total number of simulations
    GDPICORR = sp.array(GDPIcorr)
    gdpimean = sp.mean(GDPICORR)
    gdpistdev = sp.std(GDPICORR)
    GDPCCORR = sp.array(GDPCcorr)
    gdpcmean = sp.mean(GDPCCORR)
    gdpcstdev = sp.std(GDPCCORR)
    sp.savetxt('GDPICORR.csv',GDPICORR)
    sp.savetxt('GDPCCORR.csv',GDPCCORR)
    print "The mean and standard deviation between GDP and"
    print "Investment and GDP and Consumption followed by"
    print "The lists of each correlation coefficient for"
    print "each series are saved in csv files"
    #return gdpimean, gdpistdev, gdpcmean, gdpcstdev

    #Calculate the mean and standard deviation of these moments
    #across the total number of simulations
    GDPAUTO = sp.array(GDPauto)
    gdpsimmean = sp.mean(GDPAUTO)
    gdpsimstdev = sp.std(GDPAUTO)
    IAUTO = sp.array(Iauto)
    isimmean = sp.mean(IAUTO)
    isimstdev = sp.std(IAUTO)
    CAUTO = sp.array(Cauto)
    csimmean = sp.mean(CAUTO)
    csimstdev = sp.std(CAUTO)
    sp.savetxt('GDPAUTO.csv',GDPAUTO)
    sp.savetxt('IAUTO.csv',IAUTO)
    sp.savetxt('CAUTO.csv',CAUTO)
    print "GDP/Investment/Consumption Simulations Mean/Standard Deviation"
    print "of Autocorrelation. The Autocorrelation Coefficients"
    print "of GDP,Investment,Consumption for each series have been saved"
    print "separately in csv files"
    #return gdpsimmean, gdpsimstdev, isimmean, isimstdev, csimmean, csimstdev


    #Mean and standard deviation of the mean of each simulation
    GDPmean_Mean=sp.mean(GDPmean)
    Imean_Mean=sp.mean(Imean)
    Cmean_Mean=sp.mean(Cmean)
    GDPmean_Std=sp.std(GDPmean)
    Imean_Mean_Std=sp.std(Imean)
    Cmean_Mean_Std=sp.std(Cmean)


    #Mean and standard deviation of the standard deviation of each simulation
    GDPstd_Mean=sp.mean(GDPstd)
    Istd_Mean=sp.mean(Istd)
    Cstd_Mean=sp.mean(Cstd)
    GDPstd_Std=sp.std(GDPstd)
    Istd_Std=sp.std(Istd)
    Cstd_Std=sp.std(Cstd)

    #mean for particular time periods
    GDP_mean = sp.zeros((T-1))
    invest_mean = sp.zeros((T-1))
    consumption_mean =sp.zeros((T-1))
    for j in range(0,T-1): #don't use first and last periods
        GDP_mean[j] = mean(GDP[j,:])
        invest_mean[j] = mean(I[j,:])
        consumption_mean[j] = mean(C[j,:])

    #return GDP_mean,invest_mean,consumption_mean

    #might need to chop off first and last periods
    GDP_sort=sp.sort(GDP)
    invest_sort=sp.sort(I)
    consumption_sort=sp.sort(C)


    lowerBound=.05*float(reps)
    lowerBound= lowerBound-lowerBound%1 #5%

    upperBound=reps-lowerBound #95%
    #lowerBound=0
    #upperBound=4

    GDP_lower=GDP_sort[:,lowerBound]
    GDP_upper=GDP_sort[:,upperBound]
    invest_lower=invest_sort[:,lowerBound]
    invest_upper=invest_sort[:,upperBound]
    consumption_lower=consumption_sort[:,lowerBound]
    consumption_upper=consumption_sort[:,upperBound]

    #plot mean, and 90% confidence bands for GDP,I,C
    if full==1:
        start=0
    else:
        start=T/full

    x = np.arange(0,T/full-1,1)
    # Generate GDP Plot
    plt.figure(1)
    plt.plot(x,GDP_lower[start:-1], label = 'Lower Band') # ll =
    plt.plot(x,GDP_upper[start:-1], label = 'Upper Band') # mm =
    plt.plot(x,GDP_mean[start:], label = 'GDP') # aa =
    plt.title('GPD Time Series with 90% Confidence Bands')
    plt.legend(loc=5)
    plt.xlabel('time')
    plt.ylabel('GDP')
    plt.show()

    # Generate Investment Plot
    plt.figure(2)
    plt.plot(x,invest_lower[start:-1], label = 'Lower Band')
    plt.plot(x,invest_upper[start:-1], label = 'Upper Band')
    plt.plot(x,invest_mean[start:], label = 'Investment')
    plt.title('Investment Time Series with 90% Confidence Bands')
    plt.legend(loc=5)
    plt.xlabel('time')
    plt.ylabel('Investment')
    plt.show()

    # Generate Consumption Plot
    plt.figure(3)
    plt.plot(x,consumption_lower[start:-1], label = 'Lower Band')
    plt.plot(x,consumption_upper[start:-1], label = 'Upper Band')
    plt.plot(x,consumption_mean[start:], label = 'Consumption')
    plt.title('Consumption Time Series with 90% Confidence Bands')
    plt.legend(loc=3)
    plt.ylabel('Consumption')
    plt.xlabel('time')
    plt.show()


    print 'Mean of correlation between gdp and investment',gdpimean
    print 'Standard Deviation of correlation between gdp and investment',gdpistdev

    print

    print 'Mean of correlation between gdp and consumption',gdpcmean
    print 'Standard Deviation of correlation between gdp and consumption',gdpcstdev

    print

    print 'Mean of autocorrelation of gdp',gdpsimmean
    print 'Standard Deviation of autocorrelation of gdp',gdpsimstdev

    print

    print 'Mean of autocorrelation of investment',isimmean
    print 'Standard Deviation of autocorrelation of investment',isimstdev

    print

    print 'Mean of autocorrelation of consumption',csimmean
    print 'Standard Deviation of autocorrelation of consumption',csimstdev

    print

    print 'Mean of GDP',GDPmean_Mean
    print 'Standard Deviation of GDP',GDPmean_Std

    print

    print 'Mean of investments',Imean_Mean
    print 'Standard Deviation of investment',Imean_Mean_Std

    print

    print 'Mean of consumption',Cmean_Mean
    print 'Standard Deviation of consumption',Cmean_Mean_Std

    print

    print 'Mean of standard deviation of GDP',GDPstd_Mean
    print 'Standard Deviation of the standard deviation of GDP',GDPstd_Std

    print

    print 'Mean of standard deviation of investment',Istd_Mean
    print 'Standard Deviation of the standard deviation of investment',Istd_Std

    print

    print 'Mean of standard deviation of consumption',Cstd_Mean
    print 'Standard Deviation of the standard deviation of consumption',Cstd_Std

    return GDP, I, C