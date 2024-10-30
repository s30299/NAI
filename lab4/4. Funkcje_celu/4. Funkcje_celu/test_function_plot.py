import math
import sys
import numpy as np
import matplotlib.pyplot as plt

# Test functions
def empty(x, y): #0
    return x + y
def emptyAckley(x,y): #1
    a = -20*np.exp(-0.2*np.sqrt((x*x+y*y)))
    b = -np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.e+20
    return a-b

def emptyBooth(x, y): #2
    return (x+2*y-7)*(x+2*y-7)+(2*x+y-5)*(2*x+y-5)
def emptyBeale(x, y): #3
    return (1.5-x+x*y)*(1.5-x+x*y)+(2.25-x+x*y*y)*(2.25-x+x*y*y)+(2.625-x+x*y*y*y)*(2.625-x+x*y*y*y)
def emptyMatyas(x, y): #4
    return 0.26*(x*x+y*y)-0.48*x*y

def emptyStybylskiFang(x, y): #5
    return (x*x*x*x-16*x*x+5*x)/2+(y*y*y*y-16*y*y+5*y)/2

def plot_function(func, xlim=(-5, 5), ylim=(-5, 5)):
    x = np.linspace(xlim[0], xlim[1], 400)
    y = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
def choice(x):
    if(x==0 or x=='Empty'):
        return empty
    if(x==1 or x=='Ackley'):
        return emptyAckley
    if(x==2 or x=='Booth'):
        return emptyBooth
    if(x==3 or x=='Beale'):
        return emptyBeale
    if(x==4 or x=='Matyas'):
        return emptyMatyas
    if(x==5 or x=='StybylskiFang'):
        return emptyStybylskiFang

if __name__ == "__main__":
    if(len(sys.argv)>1):
        option=(sys.argv[1])
    else:
        print("0. empty")
        print("1. Ackley")
        print("2. Booth")
        print("3. Beale")
        print("4. Matyas")
        print("5. StybylskiFang")
        option = input("Enter your choice: ")



    plot_function(choice(option))
