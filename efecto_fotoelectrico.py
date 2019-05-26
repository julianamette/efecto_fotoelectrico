'''
Programa para efecto fotoelectrico labo 5, 2019
author: julianamette3@gmail.com

'''
import numpy as np
import matplotlib.pyplot as plt 
import os
from lmfit import Model
from lmfit import Parameter
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import simu as s
import time
import simu2 as s2
from lmfit.models import SkewedGaussianModel

def gaussiana(sigma, mu, a, x):
    return a * np.exp(-0.5 * ( (x - mu) / sigma )**2)

def act(xo,a,x):
    return np.heaviside(x - xo,0)*(x -xo)*a

file_list = [f for f in os.listdir('espectro_leds') if not f.startswith('.')]
file_list_1 = [f for f in os.listdir('datos_posta') if not f.startswith('.')]


colores = {'blanco' : 'gray',
         'naranja' : 'orange',
         'amarillo_2puntos': 'yellow' ,
         'azul_punto': 'blue',
         'rojo':'red',
         'violeta': 'violet',
         'verde': 'green' ,
          'rosa_L': 'pink',
          'verde2_L' : 'darkgreen',
          }

led_esp = {}
led_volt = {}
for f in file_list:
    led_esp[os.path.splitext(f)[0]] = np.genfromtxt('espectro_leds/' + f , delimiter = ',')

for f in file_list_1:
    led_volt[os.path.splitext(f)[0]] = np.load('datos_posta/' + f)

def plot_leds(led_esp):
    for c in led_esp:
        plt.plot(led_esp[c][:,0],led_esp[c][:,1], color = colores[c],label = c)

    plt.legend()
    plt.grid()
    plt.show()

def fit_gauss(led_esp, led ):
    model = Model(gaussiana, independent_vars = ['x'])
    model.set_param_hint('sigma', value = 1, vary = True)
    model.set_param_hint('mu', value = led_esp[led][:,0][np.argmax(led_esp[led][:,1])], vary = True )
    model.set_param_hint('a', value = np.max(led_esp[led][:,1]) )
    result = model.fit(led_esp[led][:,1], x = led_esp[led][:,0], nan_policy = 'propagate')
    return result

def fit_skewedgauss(led_esp, led ):
    model = SkewedGaussianModel()
    params = model.make_params(amplitude=1, center=400, sigma=5, gamma=1)
    result = model.fit(led_esp[led][:,1],params, x = led_esp[led][:,0], nan_policy = 'propagate')
    return result

def fit_act(led_volt, led):
    model = Model(act, independent_vars = ['x'])
    model.set_param_hint('xo', value = -1, vary = True)
    model.set_param_hint('a', value = 1, vary = True )
    result = model.fit(led_volt[led][:,1], x = led_volt[led][:,0], nan_policy = 'propagate')
    return result

xos = {}
def extraccion_vo():
    for led in led_volt:
        fitteo = fit_act(led_volt,led)
        plt.plot(led_volt[led][:,0] , fitteo.best_fit)
        plt.plot(led_volt[led][:,0] , led_volt[led][:,1])
        plt.title(led)
        plt.show()
        xos[led] = fitteo.best_values['xo']

h = 4.13e-15
x = np.linspace(100,1000,5000)
y = gaussiana(30, 485, 1,x)
y = y/np.sum(y)
x = 300000e12/x
v = np.linspace(-2,2,1000)

def set(led):
    x = led_esp[led][:,0]
    y = led_esp[led][:,1]
    y = y/np.sum(y)
    x = 300000e12/x
    return x, y 

def simular(n,v,x,y,h):
    start = time.time()
    g = s.simu(n = n, v = v , phi = 2.3, probas = y, frec = x, h = h)
    stop = time.time()
    print(stop-start)
    plt.plot(v,g,'.')
    plt.show()
    return g

def simular2(n,v,x,y,h, plot = 'si'):
    start = time.time()
    g = s2.simulacion(n = n, v = v , phi = 2.3, proba = y, frec = x, h = h)
    stop = time.time()
    print(stop-start)
    plt.plot(v,g,'.-')
    if plot == 'si':
        plt.show()
    return g

#extraccion_vo()
def comparar_sim_sigma(array_sigmas):
    for i in array_sigmas:
        x = np.linspace(100,1000,5000)
        y = gaussiana(i, 485, 1,x)
        y = y/np.sum(y)
        x = 300000e12/x
        v = np.linspace(-2,2,1000)
        g = simular2(100000,v,x,y,h, plot = 'no')
    plt.show()

e = 1.602176565e-19
def derivar(led_volt,led):
    dg = np.diff(led_volt[led][:,1])/np.diff(led_volt[led][:,0]) #* (h/e) 
    plt.plot(led_volt[led][:,0][:-1],dg,label = 'derivado')
    plt.plot(led_volt[led][:,0],led_volt[led][:,1],label = 'normal')
    plt.title(led)
    plt.legend()
    plt.grid()
    plt.show()


def max_der(led):
    plt.plot(led_volt[led][:,0][:-1],np.diff(led_volt[led][:,1])*20,'.')
    plt.plot(led_volt[led][:,0],led_volt[led][:,1],'.')
    arg = np.argmax(np.diff(led_volt[led][:,1]))
    plt.plot(led_volt[led][:,0][arg],led_volt[led][:,1][arg],'r*')
    plt.show()

for led in led_volt:
    max_der(led)



