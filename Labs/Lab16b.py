# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <headingcell level=1>

# Spencer Lyon

# <headingcell level=2>

# Lab 17?

# <headingcell level=4>

# Import Statements

# <codecell>

import scipy as sp 
import scipy.sparse as spar
from scipy import linalg as la
from scipy.sparse import linalg as sparla
import matplotlib.pyplot as plt
import networkx as nx

# <headingcell level=4>

# Problem 1

# <codecell>

def connected(A):
    DA = sp.diag(sp.sum(A,1))
    QA = DA-A
    conn = sp.sort(la.eig(QA)[0])[1]>0
    return [QA, sp.sort(la.eig(QA)[0])[1] ,conn]

# <codecell>

n = 15
randMat = sp.rand(n,n)
c = [0.25, 0.5, 0.75]
x,y,z = [(randMat>Q)*1 for Q in c]
xGraph, yGraph, zGraph = [nx.to_networkx_graph(x),nx.to_networkx_graph(y),nx.to_networkx_graph(z)]
[connected(x)[2],connected(y)[2],connected(z)[2]]

# <headingcell level=5>

# Above I generated three random matricies and checked to see if they were connected. The return value is a Boolean telling wheather or not the graph is connected. As can be seen all three of them are connected so in general random matricies are connected

# <codecell>

# Just for Fun
nx.draw(xGraph)

# <codecell>

nx.draw(yGraph)

# <codecell>

nx.draw(zGraph)

# <headingcell level=4>

# Problem 2

# <codecell>

img = plt.imread('/Users/spencerlyon2/Desktop/logo2.png')
img = img[:,:,1]
height = img.shape[0]
width = img.shape[1]

# <codecell>

def weight(row,col,k,l, radius=5, sigmaI=0.02, sigmaX=3.0):
    try:
        tmp = sp.exp(-(sp.absolute(img[row,col]-img[k,l])/sigmaI**2))
    except:
        return 0

    d = sp.math.sqrt((row-k)**2+(col-l)**2)
    if d<radius:
        w = tmp*sp.exp(-d/sigmaX**2)
    else:
        w=0
        
        
    return w

# <codecell>

def ncuts():
    
    radius = 5 
    sigmaI = 0.02 
    sigmaX = 3.0 
    height = img.shape[0]
    width = img.shape[1]
    
    nodes = img.flatten()
    
    W = spar.lil_matrix((nodes.size, nodes.size),dtype=float)
    D = sp.zeros((1,nodes.size))
    
    for row in range(height):
        for col in range(width):				
            for k in range(row-radius,row+radius):
                for l in range(col-radius,col+radius):
                    try:
                        w = weight(row,col,k,l)
                        W[row*width+col,k*width+l] = w
                        D[0,row*width+col] += w		
                    except:
                        continue
                        
    D = spar.spdiags(D, 0, nodes.size, nodes.size)
    
    Q = D - W
    
    evals, evecs = sparla.eigs(W, k=2, which="SR")
    e = max(evals)
    
    return Q, e

# <codecell>

[Q,e] = ncuts()

# <codecell>

Q

# <headingcell level=4>

# Problem 3

# <codecell>

def segmented():
    
    radius = 5 
    sigmaI = 0.02 
    sigmaX = 3.0 
    height = img.shape[0]
    width = img.shape[1]
    flatImg = img.flatten()
    darkImg = flatImg
    brightImg = flatImg
    
    nodes = img.flatten()
    
    W = spar.lil_matrix((nodes.size, nodes.size),dtype=float)
    D = sp.zeros((1,nodes.size))
    
    for row in range(height):
        for col in range(width):				
            for k in range(row-radius,row+radius):
                for l in range(col-radius,col+radius):
                    try:
                        w = weight(row,col,k,l)
                        W[row*width+col,k*width+l] = w
                        D[0,row*width+col] += w		
                    except:
                        continue
                        
    D = spar.spdiags(D, 0, nodes.size, nodes.size)

    Q = D - W
     
    D1 = D.todense()
    Q1 = Q.todense()
    
    diags = sp.diag(D1)
    DminusHalf = sp.diag(diags**-0.5)
    
    
    segQ = sp.dot(sp.dot(DminusHalf, Q1),DminusHalf)
    vals, vecs = la.eig(segQ)
    
    vecind = sp.argsort(vals)[1]
    theVec = vecs[vecind]

    for i in range(0,height**2):
        if theVec[i] < 0:
            darkImg[i] = 0.0
        else:
            brightImg[i] = 0.0
            
    
    darkImg = sp.reshape(darkImg, (height,height))
    brightImg = sp.reshape(brightImg, (height,height))
             
    
    
    
    return darkImg, flatImg, brightImg

# <codecell>

darkImg, flatImg, brightImg = segmented()

# <codecell>

plt.gray()
plt.imshow(img)

# <codecell>

plt.imshow(darkImg)

# <codecell>

plt.imshow(brightImg)

# <codecell>

plt.imshow(darkImg + brightImg)

