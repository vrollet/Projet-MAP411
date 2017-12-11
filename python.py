import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import sparse 

#Paramètres du modèle

a_max = 2
T_max = 2*a_max
N_a = 120
N_t = 50
Dt = T_max/N_t
Da = a_max/N_a
CFL = Dt/Da

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
    B = sparse.diags([-cfl, 1+cfl+Dt*mu(Da*i)], [-1, 0], shape=(N_a, N_a), format = 'csc')
    #On trouve l'inverse de B
    B_I = scipy.sparse.linalg.inv(B)
    #On implémente le schéma
    #On donne une valeur initiale pour P
    ai = [i*a_max/N_a for i in range(N_a)]
    P0=np.transpose(np.mat([zeta(i) for i in ai]))
    #On garde en mémoire toutes les valeurs de P 
    P=list()
    P.append(P0)
    for i in range(N_t):
        P.append(B_I*P[-1])
    #On affiche P à différents temps
    x=[i*a_max/N_a for i in range(N_a)]
    y0 = np.transpose(P[0])
    y1 = np.transpose(P[10])
    y2 = np.transpose(P[20])
    y3 = np.transpose(P[30])
    y4 = np.transpose(P[40])
    plt.axis([0, a_max, 0, 10])
    plt.plot(x,y0, label='initial')
    plt.plot(x,y1, label='après 10dt')
    plt.plot(x,y2, label='après 20dt')
    plt.plot(x,y3, label='après 30dt')
    plt.plot(x,y4, label='après 40dt')
    plt.legend()
    plt.show()
    return


draw_beta_mu()
question2()
