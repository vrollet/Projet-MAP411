import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import scipy
from scipy import integrate as integr
from scipy import optimize as opti
from scipy import sparse as sparce
from mpl_toolkits.mplot3d import Axes3D


#Paramètres du modèle

a_max = 2
T_max = 2*a_max
N_a = 120
N_t = 50
Dt = T_max/N_t
Da = a_max/N_a
CFL = Dt/Da
l = math.pi
N_x = 100
h = l/(N_x+1)

def beta(a):
    return 10*a*(a_max - a)*math.exp(-20*(a-(a_max/3))**2)

def mu(a):
    if a==a_max: 
        return 9999999999999999 #Pour éviter des erreurs numériques dues à la divergence de mu en a_max
    else:
        return 1/(a_max-a)

def zeta(a):
    return math.exp(-30*(a-(a_max/4))**2)

def zeta2(a):
    lambda_e=1.24025
    return ((a_max-a)/a_max)*np.exp(-lambda_e*a)

def zeta3(a):
    return math.sqrt(a_max**2-(3/4)*(a**2))-(a_max/2)

def zeta4(a):
    if a<3*a_max/4:
        if a>a_max/4:
            return 1
    return 0

def pi(x):
    if x==a_max:
        return 0.0000000001 #Pour éviter une erreur d'intégration due à la fonction mu divergente en a_max
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

def draw_pi():
    resolution = 0.005
    x = [i*resolution for i in range(int(a_max/resolution))]
    y = [pi(a) for a in x]   
    plt.axis([0, a_max, 0, 1.1])
    plt.plot(x,y, label='pi')
    plt.legend()
    plt.show()
    return

def draw_zeta():
    resolution = 0.005
    x = [i*resolution for i in range(int(a_max/resolution))]
    y = [zeta(a) for a in x]   
    plt.axis([0, a_max, 0, 1.1])
    plt.plot(x,y, label='zeta')
    plt.legend()
    plt.show()
    return

def draw_m():
    resolution = 0.005
    x = [i*resolution for i in range(int(a_max/resolution))]
    y = [m(a) for a in x]   
    plt.axis([0, a_max, 0, 6.1])
    plt.plot(x,y, label='m')
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
     
    #On trace p(a,x) 
    Z=list()
    for i in range(len(P[0])):
        Z.insert(0,[])
        for j in range(len(P)):
            Z[0].append(np.array(P[j][i])[0][0])
    
    f, ax = plt.subplots()
    ax.set_title('Evolution de p en fonction du temps')
    ax.imshow(Z,interpolation='bilinear', extent = (0,T_max,0,a_max))
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

def matriceA():
    #Cette fonction implémente la matrice A
    #On commence par implémenter A1:
    A1 = np.zeros((N_a,N_a))
    for i in range(1,N_a):
        A1[i,i-1]=1
    
    M=[m(Da*i) for i in range(N_a+1)]
    W=[Da/2]+[Da for i in range(N_a-1)]+[Da/2]
    for i in range(N_a):
        A1[0,i]=W[i+1]*M[i+1]
    
    #Puis A
    lambda_e=1.24025
    A = (-np.eye(N_a)+A1)/Da- lambda_e*np.eye(N_a)
    
    A=np.matrix(A)
    return A

def question14():

    A = matriceA()
    #On cherche les valeurs propres de A
    vp_A = np.linalg.eigvals(A)
    
    #On vérifie qu'elles respectent l'équation trouvée :
#    l=list()
#    for vp in vp_A:
#        alpha = 1+Da*(lambda_e+vp)
#        rep = math.pow(alpha,N_a)
#        for i in range(1, N_a+1):
#            rep+=W[i]*M[i]*math.pow(1+ Da*(lambda_e+vp),N_a-i)
#        l.append(rep)
#    
    #Dans la liste l, on ne trouve que des valeurs très petites.

    #Enfin, on implémente B et B^-1
    B=np.matrix(np.eye(N_a))-Dt*A    
    B_1=np.linalg.inv(B)
    
    #On regarde les valeurs propres de B_1
    vp_B = np.linalg.eigvals(B)
    vp_B = [abs(i) for i in vp_B]
    min_vp=min(vp_B)
    #print(min_vp)
    
    
    return B_1


def u_to_p(u,t):
    lambda_e=1.24025
    #Prend un vecteur u en entrée et un temps t, et renvoie le vecteur p associé
    p=list()
    for i in range(N_a):
        a=Da*(i+1)
        p.append(pi(a)*np.exp(lambda_e*t)*u[i])
    return p

def p_to_u(p,t):
    lambda_e=1.24025
    #Prend un vecteur p en entrée et un temps t, et renvoie le vecteur u associé
    u=list()
    for i in range(N_a):
        a=Da*(i+1)
        u.append((1/pi(a))*np.exp(-lambda_e*t)*p[i])
    return u


def question15():

    #On implémente le schéma 
    B_1=question14()
    
    ##PREMIERE CONDITION INITIALE
    
    #On convertit la condition initiale p en u
    p=[zeta2(i*Da) for i in range(1,N_a+1)]
    u=p_to_u(p,0)

    
    #On applique la matrice B_1 a u pour passer d'un instant t à l'instant t+1
    res_u = list()
    res_u.append(u)
    
    for i in range(N_t):
        #Calcul de u à t+1
        u=B_1*np.transpose(np.mat(res_u[-1]))
        #Conversion en liste
        u=list(np.array(np.transpose(u))[0])
        res_u.append(u)
    
    #On repasse en p
    res_p=list()
    for i in range(N_t):
        res_p.append(u_to_p(res_u[i], Dt*i))
    
    #On affiche le résultat

    x=[i*a_max/N_a for i in range(N_a)]
    y0 = res_p[0]
    y1 = res_p[5]
    y2 = res_p[10]
    y3 = res_p[15]
    plt.axis([0, a_max, 0, 1.2])
    plt.plot(x,y0, label='initial')
    plt.plot(x,y1, label='après 5dt')
    plt.plot(x,y2, label='après 10dt')
    plt.plot(x,y3, label='après 15dt')
    plt.legend()
    plt.show()

    ##SECONDE CONDITION INITIALE
    
    #On convertit la condition initiale p en u
    p=[zeta(i*Da) for i in range(N_a+1)]
    u=p_to_u(p,0)
    
    #On applique la matrice B_1 a u pour passer d'un instant t à l'instant t+1
    res_u = list()
    res_u.append(u)
    
    for i in range(N_t):
        #Calcul de u à t+1
        u=B_1*np.transpose(np.mat(res_u[-1]))
        #Conversion en liste
        u=list(np.array(np.transpose(u))[0])
        res_u.append(u)
    
    #On repasse en p
    res_p=list()
    for i in range(N_t):
        res_p.append(u_to_p(res_u[i], Dt*i))
    
    #On affiche le résultat

    x=[i*a_max/N_a for i in range(N_a)]
    y0 = res_p[0]
    y1 = res_p[5]
    y2 = res_p[10]
    y3 = res_p[15]
    plt.axis([0, a_max, 0, 11.2])
    plt.plot(x,y0, label='initial')
    plt.plot(x,y1, label='après 5dt')
    plt.plot(x,y2, label='après 10dt')
    plt.plot(x,y3, label='après 15dt')
    plt.legend()
    plt.show()
    
    #Proposition de condition initiale
    print("On propose la condition initiale suivante :")
    y0 = [zeta3(i*Da) for i in range(N_a)]
    plt.plot(x,y0, label='initial')
    plt.axis([0, a_max, 0, 1.2])
    plt.legend()
    plt.show()
    
    print("On a alors les résultats :")
    ##PROPOSITION DE CONDITION INITIALE
    
    #On convertit la condition initiale p en u
    p=[zeta3(i*Da) for i in range(N_a+1)]
    u=p_to_u(p,0)
    
    #On applique la matrice B_1 a u pour passer d'un instant t à l'instant t+1
    res_u = list()
    res_u.append(u)
    
    for i in range(N_t):
        #Calcul de u à t+1
        u=B_1*np.transpose(np.mat(res_u[-1]))
        #Conversion en liste
        u=list(np.array(np.transpose(u))[0])
        res_u.append(u)
    
    #On repasse en p
    res_p=list()
    for i in range(N_t):
        res_p.append(u_to_p(res_u[i], Dt*i))
    
    #On affiche le résultat

    x=[i*a_max/N_a for i in range(N_a)]
    y0 = res_p[0]
    y1 = res_p[5]
    y2 = res_p[10]
    y3 = res_p[15]
    plt.axis([0, a_max, 0, 11.2])
    plt.plot(x,y0, label='initial')
    plt.plot(x,y1, label='après 5dt')
    plt.plot(x,y2, label='après 10dt')
    plt.plot(x,y3, label='après 15dt')
    plt.legend()
    plt.show()
    
    #Vérification du modèle
    print("On vérifie le modèle avec la condition initiale suivante :")
    y0 = [zeta4(i*Da) for i in range(N_a)]
    plt.plot(x,y0, label='initial')
    plt.axis([0, a_max, 0, 1.2])
    plt.legend()
    plt.show()
    
    print("On a alors les résultats :")
    ##PROPOSITION DE CONDITION INITIALE
    
    #On convertit la condition initiale p en u
    p=[zeta4(i*Da) for i in range(N_a+1)]
    u=p_to_u(p,0)
    
    #On applique la matrice B_1 a u pour passer d'un instant t à l'instant t+1
    res_u = list()
    res_u.append(u)
    
    for i in range(N_t):
        #Calcul de u à t+1
        u=B_1*np.transpose(np.mat(res_u[-1]))
        #Conversion en liste
        u=list(np.array(np.transpose(u))[0])
        res_u.append(u)
    
    #On repasse en p
    res_p=list()
    for i in range(N_t):
        res_p.append(u_to_p(res_u[i], Dt*i))
    
    #On affiche le résultat

    x=[i*a_max/N_a for i in range(N_a)]
    y0 = res_p[0]
    y1 = res_p[5]
    y2 = res_p[10]
    y3 = res_p[15]
    plt.axis([0, a_max, 0, 4.2])
    plt.plot(x,y0, label='initial')
    plt.plot(x,y1, label='après 5dt')
    plt.plot(x,y2, label='après 10dt')
    plt.plot(x,y3, label='après 15dt')
    plt.legend()
    plt.show()
    return

def matriceK():
    #On commence par implémenter la matrice K
    
    #On prend sigma = 1
    sigma = 1
    
    K=2*np.eye(N_x)
    for i in range(N_x-1):
        K[i][i+1]=1
        K[i+1][i]=1
    K=(sigma/(h**2))*K
    
    return np.matrix(K)

def question16():
        
    vp_K = np.linalg.eigvals(matriceK())
    vp_A = np.linalg.eigvals(matriceA())
    
    vp=list()
    for i in vp_A:
        for j in vp_K:
            vp.append(np.abs(1-Dt * (i-j)))
    
    vp_min=min(vp)
    
    return vp_min

def inverse_B():
    #Cette fonction nous fournit B_1 qui est nécéssaire à l'implémentation du schéma
    
    #On commence par implémenter la matrice B
    Ix = np.matrix(np.eye(N_x))
    Ia = np.matrix(np.eye(N_a))
    A = matriceA()
    K = matriceK()
    B = np.matrix(np.eye(N_a*N_x)- Dt*(-np.kron(Ia, K) + np.kron(A, Ix)))
    
    #On l'inverse
    B_1= np.linalg.inv(B)
    
    return B_1

def p0(a,x):
    #Donne la condition initiale de la question 17
    lambda_e=1.24025
    return ((a_max-a)/a_max)*np.exp(-lambda_e*a)*math.sin(x)

def condition_initiale():
    #Donne le vecteur p associé à la condition initiale de la question 17
    p=list()
    for i in range(N_a):
        for j in range(N_x):
            a=i*Da
            x=j*h
            p.append(p0(a,x))
    return p

def affiche3D(P):
    #Cette fonction permet d'afficher en 3D un vecteur P

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Make data.
    Y = np.arange(0, a_max, Da)
    X = np.arange(0, np.pi-h, h)
    X, Y = np.meshgrid(X, Y)
    Z=[]
    for i in range(N_a):
        l=[]
        for j in range(N_x):
            l.append(P[N_x*i+j])
        Z.append(l)
    Z=np.array(Z)
    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(0, 1)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('âge')
    ax.set_zlabel('Densité')
    
    plt.show()

    return

def u_to_p2(u,t):
    lambda_e=1.24025
    #Prend un vecteur u en entrée et un temps t, et renvoie le vecteur p associé
    p=list()
    for i in range(N_t*N_a):
        a=Da*(i+1)
        p.append(pi(a)*np.exp(lambda_e*t)*u[i])
    return p

def p_to_u2(p,t):
    lambda_e=1.24025
    #Prend un vecteur p en entrée et un temps t, et renvoie le vecteur u associé
    u=list()
    for i in range(N_t*N_a):
        a=Da*(i+1)
        u.append((1/pi(a))*np.exp(-lambda_e*t)*p[i])
    return u

def question17():
    print("On a la condition initiale suivante :")
    P=condition_initiale()
    affiche3D(P)
    
    U=p_to_u2(P,0)
    B_1=inverse_B()
    
    print("Après un temps de 0.4 :")
    for i in range(5):
        U=np.matrix.transpose(np.matrix(U))
        U=B_1*U
    Ua=np.array(np.matrix.transpose(U))
    P=u_to_p2(Ua,0.4)
    affiche3D(P)
    return


