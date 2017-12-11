import matplotlib.pyplot as plt
import numpy as np
import math

#Paramètres du modèle

a_max = 2
T_max = 2*a_max
N_a = 120
N_t = 50

def beta(a):
    return 10*a*(a_max - a)*math.exp(-20*(a-(a_max/3))**2)

def mu(a):
    return 1/(a_max-a)

def zeta(a):
    return math.exp(-30*(a-(a_max/4))**2)

def draw_beta_mu():
    
    resolution = 0.005
    
    x = [i*resolution for i in range(int(a_max/resolution))]
    y = [beta(a) for a in x]   
    y2 = [mu(a) for a in x]
    plt.axis([0, a_max, 0, 10])
    plt.plot(x,y, label='beta')
    plt.plot(x,y2, label = 'mu')
    plt.legend()
    plt.show()
    
    return


def question2():
    #On implémente le schéma implicite décentré amont proposé
    #On commence par définir la matrice B telle que B P_(n+1) = P_n
    #On trouve l'inverse de B
    #On implémente le schéma



draw_beta_mu()
