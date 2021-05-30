'''
Written By: Abhijeet Kulkarni (amkulk@udel.edu)
For: Research Project MEEG678
Title: Model Predictive Contouring Control of autonomous vehicle.
'''
import numpy as np
import math


### bezier polynomial
def BernsteinPol(i,n,t):
    if (i < 0) or (i > n):
        return np.zeros_like(t)
    else:
        return (math.factorial(n)/(math.factorial(i)*math.factorial(n-i)))*((1-t)**(n-i))*t**i

def GetBezierOut(coefficients, t,returnderivative = False,return2derivative = False):# not vectorized for coefficients
    n = len(coefficients)-1 #degree
    B = np.float64(np.zeros_like(t))
    dB = B
    ddB = B
    for i in range(0,n+1):
        B += coefficients[i]*BernsteinPol(i,n,t)
        if returnderivative:
            dB += coefficients[i]*n*(BernsteinPol(i-1,n-1,t) - BernsteinPol(i,n-1,t))
        if return2derivative:
            ddB += coefficients[i]*n*(n-1)*(BernsteinPol(i,n-2,t) - 2*BernsteinPol(i-1,n-2,t) + BernsteinPol(i-2,n-2,t))
    if returnderivative and return2derivative:
        return B,dB,ddB
    elif returnderivative:
        return B,dB
    elif return2derivative:
        return B,ddB
    else:
        return B


### Projecting onto curve
# https://pomax.github.io/bezierinfo/

def distance(P1,P2):
    return np.sqrt((P1[0]-P2[0])**2 + (P1[1]-P2[1])**2)

def findclosest(coefMatrix,p,numSteps,t0,t1):
    d = 9999999999
    tc = 0
    LUT_length = 20
    for ti in np.linspace(t0,t1,LUT_length):

        coordinate = [GetBezierOut(coefMatrix[0,:],ti),GetBezierOut(coefMatrix[1,:],ti)]    
        q = distance(coordinate, p)
        if q < d:
            d = q
            tc = ti
    return tc

def ProjectOnBCurveXY(coefMatrix,p,ParaThreshold = 0.01): #accepts coefficient matrix for x and y; top row for x
    IniSearch_Length = 10
    tc = findclosest(coefMatrix,p,IniSearch_Length,0,1) # for better initial guess
    tc_last = 0
    step = 1/IniSearch_Length
    while np.abs(tc_last-tc)>ParaThreshold:
        tc_last = tc
        step = step/2
        tc = findclosest(coefMatrix,p,5,tc_last-2*step,tc_last+2*step) # keep searching in a window of 5 points
    if tc>1:
        tc = 1
    if tc<0:
        tc = 0
    return tc

def findclosestONTrack(X,Y,xt,yt):
    ex = xt - X
    ey = yt - Y
    indx = np.argmin(ex**2 + ey**2)
    return indx

#####################
#curve fitting for bezier
def binomial(n, k):
    return math.factorial(n) / math.factorial(k) / math.factorial(n - k)

def generateBasisMatrix(degree):
    n = degree + 1
    M = (np.zeros([n,n]))

    #populate the main diagonal
    k = n - 1
    
    for i in range(0,n):
        M[i, i] =  binomial(k, i)

    #compute the remaining values
    for c in range(0,n):
        for r in range(c+1,n):
            sign = 1 if ((r+c)%2==0) else -1
            value = binomial(r, c) * M[r,r]
            M[r, c] =  sign * value

    return M

def formTMatrix(t_points, degree):
    n=degree+1
    numPoints = len(t_points)
    T = np.matrix([[t**i for i in range(0,n)] for t in t_points])
    return T

def fitCurvetopoints(points,t_points,degree):
    T = formTMatrix(t_points,degree)
    M = generateBasisMatrix(degree)
    C = np.asarray(np.linalg.inv(M)@np.linalg.pinv(T)@points).flatten()
    return C
#################


def GenOffsetTrack(xt,yt,dist):       #generation of track at an offset to the input track
    maxidx = len(xt)
    x_out,y_out = np.zeros_like(xt),np.zeros_like(yt)
    for indx in range(maxidx):
        x1,y1 = xt[indx],yt[indx]
        x2,y2 = xt[(indx+1)%maxidx],yt[(indx+1)%maxidx]
        x_vec,y_vec = x2-x1,y2-y1
        length = np.sqrt(x_vec**2+y_vec**2)
        x_out[indx],y_out[indx] = x1-(y_vec/length)*dist[indx],y1+(x_vec/length)*dist[indx] # rotate unit vector at 90 deg and traverse dist
    return x_out,y_out


def GetTrackBoundsConstants(Xcar,Ycar,xc,yc,xi,yi,xo,yo): #(point,centertrack,parallel track)
    # a*x+b*y+c>0
    # dir((x1-x2)y-(y1-y2)x-y1(x1-x2)+x1(y1-y2))>=0
    safetydist = 0.3 #tunable
    indx = findclosestONTrack(Xcar,Ycar,xc,yc)
    maxidx=len(xc)

    # inner track
    x1,y1 = xi[indx],yi[indx]
    x2,y2 = xi[(indx+1)%maxidx],yi[(indx+1)%maxidx]
    a_i = -(y1-y2)
    b_i = x1-x2
    c_i = -y1*(x1-x2)+x1*(y1-y2)-safetydist

    dir_i = np.sign(a_i*xo[indx]+b_i*yo[indx]+c_i)
    
   # outer track
    x1,y1 = xo[indx],yo[indx]
    x2,y2 = xo[(indx+1)%maxidx],yo[(indx+1)%maxidx]
    a_o = -(y1-y2)
    b_o = x1-x2
    c_o = -y1*(x1-x2)+x1*(y1-y2)+safetydist

    dir_o = np.sign(a_o*xi[indx]+b_o*yi[indx]+c_o)

                # a                     b                              c
    return np.array([[dir_i*a_i]]),np.array([[dir_i*b_i]]),np.array([[dir_i*c_i]]),np.array([[dir_o*a_o]]),np.array([[dir_o*b_o]]),np.array([[dir_o*c_o]])


