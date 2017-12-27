import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy import integrate as integr
from scipy import optimize as opti
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
    if a==a_max: 
        return 9999999999999999
    else:
        return 1/(a_max-a)

def zeta(a):
    return math.exp(-30*(a-(a_max/4))**2)

def pi(x):
    if x==a_max:
        (C,err)=integr.quad(mu,0,x*0.999)
    else:
        (C,err)=integr.quad(mu,0,x)
    return np.exp(-C) 

def m(a):
    return beta(a)*pi(a)

def F(x):
    #Cette fonction implémente F-1, dont on cherche un zéro
    F = -1 + 0.5*beta(0)*pi(0) + 0.5*beta(a_max)*pi(a_max*0.999999999999)*np.exp(-x*a_max)
    for i in range(1,N_a):
        F += beta(i*Da)*np.exp(-x*i*Da)*pi(i*Da)
    return F
    

def draw_beta_mu():
    #Cette fonction permet de tracer beta et mu pour observer l'allure de ces deux courbes
    
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
    Bmaindiag = [1+CFL+Dt*mu(Da*i) for i in range(N_a)]
    Bsubdiag = [-CFL for i in range(1,N_a)]
    Bdiagonals = [Bmaindiag,Bsubdiag]
    B=sparse.diags(Bdiagonals, [0, -1],format='csc')
    #On trouve l'inverse de B
    B_I = sparse.linalg.inv(B)
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
    y0 = P[0]
    y1 = P[5]
    y2 = P[10]
    y3 = P[15]
    y4 = P[20]
    plt.axis([0, a_max, 0, 1.2])
    plt.plot(x,y0, label='initial')
    plt.plot(x,y1, label='après 5dt')
    plt.plot(x,y2, label='après 10dt')
    plt.plot(x,y3, label='après 15dt')
    plt.plot(x,y4, label='après 20dt')
    plt.legend()
    plt.show()
    return

def question10():
    #On implémente la résolution par un algorithme de Newton et on la compare
    #avec la solution proposée par Python
    
    #Paramètres de la méthode de Newton
    x0=1 
    epsilon=1e-6
    h=1e-4

    #Algorithme 
    x = x0 # Initialisation
    # while abs(F(x)) > epsilon:
    for i in range (1,11):
        derivee = (F(x+h) - F(x)) / h # dérivée en x_k par taux d’accroissement
        x = x - F(x)/derivee # nouvelle valeur x_{k+1} de x
        
    print("x =", x)
    #Comparaison des résultats
    verif = opti.fsolve(F,1)
    print("verif = ", verif)
    return (x,verif) 

def question14():
    #On implémente numériquement la matrice A
    ai=[i*Da for i in range(N_a+1)]
    w=[0.5]+[1 for i in range(1,N_a)]+[0.5]
    lambda_e=8.5716
    alpha_tilde=2+Dt*lambda_e
    alpha = (1-alpha_tilde)/Dt
    c=[(1+m(ai[1]))/Dt]+[(1+w[i+1]*m(ai[i+1])-alpha_tilde*w[i]*m(ai[1]))/Dt for i in range(1,N_a)]+[(1-0.5*m(ai[N_a])*alpha_tilde)/Dt]
    
    A=np.zeros((N_a+1,N_a+1))
    A[0]=c
    for i in range(1,N_a+1):
        A[i][i-1]=-1
        A[i][i]=alpha
    
    #On affiche les valeurs propres de A
    print("Les valeurs propres de A sont : ")
    vpA=scipy.linalg.eigvals(A)
    print(vpA)
    
    #On peut vérfier que les valeurs propres vérifient l'équation (7) dicrétisée
    res=list()
    for l in vpA:
        r=0
        for i in range(N_a+1):
            r+=Da*w[i]*m(ai[i])*np.exp(-l*ai[i])
        res.append(r)
    
    res_norme=[abs(r) for r in res]              
    
    #On implémente la matrice B
    B=np.eye(N_a+1)-Dt*A
    #On cherche le rayon spectral de B
    vp=scipy.linalg.eigvals(A)
    vp_norme=[np.abs(v) for v in vp]
    rayon = max(vp_norme)
    
    print("Le rayon spectral de B est : "+str(rayon))
    
    return
