import math
import numpy as np
from matplotlib import pyplot
import pandas as pd
from math import radians, cos, sin, tan, sqrt, fmod, atan, degrees
import datetime
from scipy import optimize as op

def MarsEquantModel(c,r,e1,e2,z,s,times,oppositions):
    times=np.array(times)

    theta=np.zeros(times.shape)
    for i in range(1,times.shape[0]):
        theta[i]=times[i]*s

    alpha = theta + z + np.degrees(np.arcsin( (e1/r*np.sin(np.radians(e2-theta-z))) - (1/r*np.sin(np.radians(c-theta-z)))) )
    errors = abs(np.degrees(np.arctan2(r*np.sin(np.radians(alpha)) + np.sin(np.radians(c)) , r*np.cos(np.radians(alpha)) + np.cos(np.radians(c)) ))%360 - oppositions%360)

    maxError = errors.max()

    return errors, maxError


def bestOrbitInnerParams(r,s,times,oppositions):
    maxError = 359
    for c_ in np.arange(148.5,149.5,0.11):
        for e1_ in np.arange(1.5,1.8,0.011):
            for e2_ in np.arange(147.8,149.3,0.11):
                for z_ in np.arange(55.75,56,.055):
                    e,mxE = MarsEquantModel(c_,r,e1_,e2_,z_,s,times,oppositions)
                    if mxE<maxError:
                        maxError=mxE
                        errors=e
                        c,e1,e2,z=c_,e1_,e2_,z_

    return c, e1, e2, z, errors, maxError



def bestMarsOrbitParams(times,oppositions):
    maxError = 359
    for r in np.arange(8,10,0.04):
        for s in np.arange(0.52406,0.52409,0.000005):
            c,e1,e2,z,err,mxE= bestOrbitInnerParams(r,s,times,oppositions)
            if mxE<maxError:
                maxError=mxE
                errors=err
                R = r
                S = s
                C = c
                E1= e1
                E2= e2
                Z = z
    return R, S, C, E1, E2, Z, errors, maxError


def bestS(r,times,oppositions):
    maxError = 360 
    for s in np.arange(0.52400,0.52410,0.000001):
        c,e1,e2,z,a,b= bestOrbitInnerParams(r,s,oppositions)
        if b<maxError:
            maxError=b
            errors=a
            S=s
    return S, errors, maxError


def bestR(s,oppositions):
    maxError = 360
    for r in np.arange(8,12,0.1):
        c,e1,e2,z,a,b= bestOrbitInnerParams(r,s,oppositions)
        if b<maxError:
            maxError=b
            errors=a
            R=r
    return R, errors, maxError


# Equant Polar to cartesian
def p2c(e1,e2):
    x = e1*cos(radians(fmod(e2,360)))
    y = e1*sin(radians(fmod(e2,360)))
    return x,y

# Intersection of Latitude and orbit.
def intersect(e1,e2,m,c,r):
    f = lambda x : [(x[0]-cos(radians(c)))**2 + (x[1]-sin(radians(c)))**2 - r**2,tan(radians(m))*(e1-x[0])-(e2-x[1])]
    res = op.fsolve(func=f,x0=[e1+2*r*cos(radians(m)),e2+2*r*sin(radians(m))])
    return res


# Plot
def plot(r,s,c,e1,e2,z,times,oppositions):
    fs=6
    alpha=0.2
    
    e1,e2=p2c(e1,e2)
#     print(f'{e1},{e2}')
    pyplot.scatter(cos(radians(c)),sin(radians(c)),c='r')
    pyplot.text(cos(radians(c))+0.3,sin(radians(c))+0.3, 'Center', fontsize=fs)
    pyplot.scatter(e1,e2,c='b')
    pyplot.text(e1+0.3,e2+0.3,'Equant',fontsize=fs)
    pyplot.scatter(0,0,c='g')
    pyplot.text(0.3,0.3,'Sun',fontsize=fs)

    i=1
    for o in oppositions:
        pyplot.arrow(0,0,1.3*r*cos(radians(o)),1.3*r*sin(radians(o)),ls='-')
        x,y = intersect(0,0,o,c,r)
        pyplot.text(x+0.3,y+0.3,f'Opp: {i}',fontsize=fs)
        pyplot.scatter(x,y,c='red',alpha=1,s=5)
        i+=1

    for o in times:
        m = fmod(o*s+z,360)
        pyplot.arrow(e1,e2,1.5*r*cos(radians(m)),1.5*r*sin(radians(m)),ls=':',alpha=0.2)
        x,y = intersect(e1,e2,m,c,r)
        pyplot.scatter(x,y,c='blue',alpha=1,s=5)

    x = np.radians(np.arange(0,360))
    pyplot.plot(cos(radians(c))+r*np.cos(x),sin(radians(c))+r*np.sin(x))
    pyplot.show()


def get_times(data):
    times = [0]

    for i in range(1,len(data)):
        intervals = datetime.datetime(data[i,0], data[i,1], data[i,2], data[i,3], data[i,4])  -  datetime.datetime(data[i-1,0], data[i-1,1], data[i-1,2], data[i-1,3], data[i-1,4])
        times+=[times[-1]+(intervals.days + intervals.seconds/(24*3600))]

    return np.array(times)


def get_oppositions(data):
    opp = np.array(data[:,5]*30 + data[:,6] + data[:,7]/60 + data[:,8]/3600)
    return opp






if __name__ == "__main__":

    # Import oppositions data from the CSV file provided
    data = np.genfromtxt(
        "../data/01_data_mars_opposition_updated.csv",
        delimiter=",",
        skip_header=True,
        dtype="int",
    )

    # Extract times from the data in terms of number of days.
    # "times" is a numpy array of length 12. The first time is the reference
    # time and is taken to be "zero". That is times[0] = 0.0
    times = get_times(data)
    assert len(times) == 12, "times array is not of length 12"

    # Extract angles from the data in degrees. "oppositions" is
    # a numpy array of length 12.
    oppositions = get_oppositions(data)
    assert len(oppositions) == 12, "oppositions array is not of length 12"

    # Call the top level function for optimization
    # The angles are all in degrees
    r, s, c, e1, e2, z, errors, maxError = bestMarsOrbitParams(
        times, oppositions
    )

    assert max(list(map(abs, errors))) == maxError, "maxError is not computed properly!"
    print(
        "Fit parameters: r = {:.4f}, s = {:.4f}, c = {:.4f}, e1 = {:.4f}, e2 = {:.4f}, z = {:.4f}".format(
            r, s, c, e1, e2, z
        )
    )
    print("The maximum angular error = {:2.4f}".format(maxError))
    

    plot(r,s,c,e1,e2,z,times,oppositions)
