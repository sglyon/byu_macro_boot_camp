import scipy as sp
import matplotlib.pyplot as plt



def ZgenImpulse(shock, RHOz,zbar,t):
	"""
	Zt = RHOz*Ztm1 + impulse
	Generates impulse technology shocks given
	a specified RHOz (=N) and shock size for
	a specified number of periods t.
	Z3 = shock
	"""
	#Generate Z for t periods, we will populate each period below
	Z = sp.zeros(t)
	#The third period the Z = shock
	Z[0] = 0
	Z[3] = shock
        #Z[T/2]=-shock
        for j in range(t):
            Z[j]=Z[j]+zbar
	return Z

def Xgen(X0,Z,PP,QQ,Xbar):
    """
    This function generates a history of X given a history
    technology shocks (Z), a P matrix, a Q matrix, and an
    intial X (X0).
    Note Xt(tilde) = PXt-1(tilde) +QZt(tilde)
    Xt=Xbar*e^Xt(tilde)
    """
    num_endog=sp.shape(PP)[1]
    T=len(Z)#sp.shape(Z)[0]
    #display(T)
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

def Ygen(Y0,Z,RR,SS):
    """
    This function generates a history of Y given a history
    technology shocks (Z), a R matrix, a S matrix, and an
    intial Y (Y0).
    Note Yt = RXt-1 +SZt
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



def impulseResponse(P, Q, R, S, N, X0, Y0, YICgen, shock, Xbar, Zbar, alpha,
                    delta, T):
    """
    Produces graphs for the impulse responses of GDP, Investments,
    and consumption.  P,Q,R,S,N are matrices that have previously
    been solved for.  Xbar and Zbar are from finding the steady
    state.  Alpha, delta, and T (number of periods) are parameters
    for the model.  The variable 'shock' is the amount of shock
    delivered to the model in the 4th time period.

    YICgen is a function specific to the model at hand that allows you to
    generate time series for GDP, investment, and consumtion given a time
    series of exogenous state variables (Z) and a time series of endogenous
    state variales (X).

    Returns vectors and plots for GDP, investments, and consumption.

    """
    #generate X,Y,Z
    Z = ZgenImpulse(shock, N,Zbar,T)
    X = Xgen(X0,Z,P,Q,Xbar)

    ##find y,i,c for each time period
    GDP,invest,consumption = YICgen(X,Z,alpha,delta)

    #Y = Ygen(Y0,Z,R,S)

    #return GDP_mean,invest_mean,consumption_mean

    #plot mean, and 90% confidence bands for GDP,I,C
    x = sp.arange(0,T-1,1)

    if shock < 0:
        direction = 'negative'
    else:
        direction = 'positive'

    # Generate IRF for GDP in response to productivity shocok
    plt.figure(1)
    plt.plot(x,GDP[:-1], label = 'GDP') # ll=
    plt.title('Impulse Response in GDP to ' + direction + ' productivity shock')
    plt.legend(loc=4)
    plt.xlabel('time')
    plt.ylabel('GDP')
    plt.show()

    # Generate IRF for Investment in response to productivity shocok
    plt.figure(2)
    plt.plot(x,invest[:-1], label = 'Investment') # mm =
    plt.title('Impulse Response in Investment to ' + direction + ' productivity shock')
    plt.legend(loc=4)
    plt.xlabel('time')
    plt.ylabel('Investment')
    plt.show()

    # Generate IRF for Consumption in response to productivity shocok
    plt.figure(3)
    plt.plot(x,consumption[:-1], label = 'Consumption') # aa =
    plt.title('Impulse Response in Consumption to ' + direction + ' productivity shock')
    plt.legend(loc=1)
    plt.xlabel('time')
    plt.ylabel('Consumption')
    plt.show()

    return GDP, invest, consumption
