# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:56:21 2021

@author: Merqueo
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
# clase de la capa de la red
#n_conn numero de conexiones
#n_neur numero de neuronas
#b vector parametro de bias tantos como numero de neuronas
#w matrz que tiene el numero de neuronas y numero de conexiones
n=#registros en el dataset
p= #caracteristicas o variables en el dataset
X,Y= #dataset
Y=Y[:,np.newaxis]
class neural_layer():
    def _init_(self, n_conn,n_neur,act_f):
        self.act_f= act_f
        self.b = np.random.rand(1,n_neur)
        self.w = np.random.rand(n_conn,n_neur)
# funciones de acttivacion
#por algoritmo de backporpagation calcular derivada
sigm =(lambda x: 1/ (1+np.e**(-x)),
       lambda x: x*(1-x))
relu =(lambda x: np.maximum(0,x))
topology=[p,4,8,16,8,2]

def create_nn(topology,act_f):
    nn=[]
    for l, layer in enumerate(topology):
        nn.append(neural_layer(topology[l],topology[l+1],act_f))
    return nn
l2_cost= (lambda Yp,Yr:np.mean((Yp-Yr)**2),
          lambda Yp,Yr:(Yp-Yr))
neural_net=create_nn(topology, sigm)
#lr taza de aprendizaje
def train(neural_net,X,Y,l2_cost,lr=0.5,train=True):
    #forward pass
    out=[(None,X)]
    for l,layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].w+neural_net.b
        a=neural_net[l].act_f[0](z)    
        out.append((z,a))
    if train:
        #backward pass
        deltas =[]
        for l reversed(range(0,len(neural_net))):
            z=out[l+1][0]
            a=out[l+1][1]
            if l== len(neural_net)-1:
                #calcular delta ultima capa
                deltas.insert(0, l2_cost[1](a,Y)*neural_net[l].act_f[1](a))
                
            else:
                #calcular delta respecto a capa previa
                deltas.insert(0, deltas[0]@ W.T*neural_net[l].act_f[1](a))
                
            W=neural_net[l].w
            #gradient descent
            neural_net[l].b=neural_net[l].b-np.mean(deltas[0],axis=0,keepdims=True)*lr
            neural_net[l].w=neural_net[l].w-out[l][1].T @ deltas[0]*lr
    return out[-1][1]
            