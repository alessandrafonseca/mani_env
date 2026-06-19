import numpy as np
import scipy.linalg
from scipy.linalg import lu

#
# Eulero in avanti
#
def eulero_avanti(f, t0, tN, y0, h):
    """Metodo di Eulero in avanti
    Input:
        f   (lambda function)               Funzione che rappresenta il termine di destra dell'ODE: f(t, y)                    
        t0  (float)                         Tempo iniziale
        tN  (float)                         Tempo finale
        y0  (float,list o numpy.ndarray)    Condizione iniziale: scalare o array.
        h   (float)                         Passo temporale

    Output:
        t_h   (numpy.ndarray)     Vettore degli istanti temporali (lunghezza N+1).
        u_h   (numpy.ndarray)     Soluzione discreta nei nodi temporali (matrice di dimensioni d x N+1).
    
    !!ATTENZIONE!!: controllare che l'output di f e il dato y0 siano vettori della stessa lunghezza!
    Note:
        Se la dimensione di y0 è 1, la soluzione è un arrey di lunghezza N+1.
    """
    # trasforma y0 in un vettore 1d
    y0 = np.atleast_1d(y0)

    # Determiniamo il numero di passi temporali N e la dimensione di y0
    N = int((tN - t0) / h)
    d = len(y0)

    # inizializzazione la matrice soluzione    
    u_h = np.zeros((d,N+1))            
    t_h = np.zeros(N+1)                 # abbiamo N passi temporali, quindi N+1 nodi
    
    # ciclo iterativo che calcola i passi di Eulero esplicito
    u_h[:,0] = y0
    t_h[0] = t0 

    for i in range(N):
        u_h[:,i+1] = u_h[:,i]+ h*f(t_h[i], u_h[:,i])
        t_h[i+1] = t_h[i]+h

    if (d==1):
       u_h = np.squeeze(u_h)

    return t_h, u_h

#
# Eulero all'indietro
#
def eulero_indietro(f, t0, tN, y0, h):
    """ Metodo di Eulero all'indietro per problemi scalari o sistemi lineari
    Input:
        f    (lambda function o numpy.ndarray) Termine di destra dell'ODE se scalare, oppure matrice A se vettoriale
        t0   (float)                           Tempo iniziale
        tN   (float)                           Tempo finale
        y0   (float, list o numpy.ndarray)     Dato iniziale (scalare o vettore)
        h    (float)                           Passo temporale

    Output:
        t_h  (numpy.ndarray)                   Vettore degli istanti temporali (lunghezza N+1).
        u_h  (numpy.ndarray)                   Soluzione discreta nei nodi temporali (matrice di dimensioni d x N+1).
    """

    # assicuro che y0 sia un vettore
    y0 = np.atleast_1d(y0)

     # Determino il numero di passi temporali
    N = int((tN - t0) / h)

    if (len(y0) == 1):      

        # inizializzazione la matrice soluzione    
        t_h = np.zeros(N+1)     # abbiamo N passi temporali, quindi N+1 nodi
        u_h = np.zeros(N+1)            

        # ciclo iterativo che calcola i passi di Eulero esplicito
        u_h[0] = y0[0]
        t_h[0] = t0 

        # parametri per il punto fisso
        nmax_pf = 300
        toll_pf = 1e-5

        for i in range(N):
            phi = lambda z: u_h[i] + h * f(t_h[i] + h, z)
            # chiamo il metodo del punto fisso
            u_pf = puntofisso(phi, u_h[i], nmax_pf, toll_pf)
            # carico il vettore u
            u_h[i+1] = u_pf[-1]
            # carico il vettore t
            t_h[i+1] = t_h[i]+h

        return t_h, u_h
    else:

        # check che f sia una matrice 
        if not isinstance(f, np.ndarray):
            raise RuntimeError('input sbagliato: f deve essere una matrice NumPy quando y0 è un vettore')

        A = f
        d = len(y0)

        # Controllo che A sia quadrata e compatibile con y0
        if A.shape[0] != A.shape[1] or A.shape[0] != d:
            raise ValueError("La matrice A deve essere quadrata e della stessa dimensione di y0")

        # inizializzazione la matrice soluzione    
        t_h = np.zeros(N+1)     # abbiamo N passi temporali, quindi N+1 nodi
        u_h = np.zeros((d,N+1))

        # ciclo iterativo che calcola (I-h*A)u^(n+1) = u^n
        u_h[:,0] = y0
        t_h[0] = t0 

        # fattorizzazione LU della matrice I-h*A
        P, L, U = lu(np.eye(A.shape[0]) - h*A)

        for i in range(N):
            u_old = u_h[:,i]
            y = fwsub(L, P.T @ u_old)
            u_h[:,i+1] = bksub(U, y)
            t_h[i+1] = t_h[i] + h

        return t_h, u_h

#
# Crank-Nicolson
#
def crank_nicolson(f, t0, tN, y0, h):
    """Metodo di Crank-Nicolson
    Input:
      f   (lambda function)   Termine di destra dell'ODE, passata come
                              funzione di tempo e spazio, f = f(t, y)
      t0  (float)             Tempo iniziale
      tN  (float)             Tempo finale
      y0  (float)             Dato iniziale
      h   (float)             Passo temporale

    Output:
        t_h   (numpy.ndarray)     t_h = vettore degli istanti in cui viene calcolata la soluzione discreta (lunghezza N)
        u_h   (numpy.ndarray)     u_h = soluzione discreta calcolata nei nodi temporali t_h 
    """
    # Determiniamo il numero di passi temporali N e la dimensione di y0
    N = int((tN - t0) / h)

    # inizializzazione la matrice soluzione    
    t_h = np.zeros(N+1)     # abbiamo N passi temporali, quindi N+1 nodi
    u_h = np.zeros(N+1)            

    # ciclo iterativo che calcola i passi di Eulero esplicito
    u_h[0] = y0
    t_h[0] = t0 

    # parametri per il punto fisso
    nmax_pf = 300
    toll_pf = 1e-5

    for i in range(N):
        phi = lambda z: u_h[i] + h * (f(t_h[i], u_h[i]) + f(t_h[i] + h, z)) / 2
        # chiamo il metodo del punto fisso
        u_pf = puntofisso(phi, u_h[i], nmax_pf, toll_pf)
        # carico il vettore u
        u_h[i+1] = u_pf[-1]
        # carico il vettore t
        t_h[i+1] = t_h[i]+h

    return t_h, u_h

#
# Lab 2: metodo del punto fisso 
#
def puntofisso(phi, x0, nmax=100, toll=1.0e-6):
    """Metodo punto fisso
    Input:
      phi   (lambda function)   Funzione di iterazione
      x0    (float)             Punto di partenza
      𝚗𝚖𝚊𝚡  (float)             Numero massimo di iterazione
      toll  (float)             Tolleranza richiesta

    Output:
      xvect (numpy.ndarray)     Vettore delle iterate.
    """

    # inizializzazione
    xvect = []
    xold = x0

    for nit in range(nmax):
        # calcolo il nuovo punto
        xnew = phi(xold)
        # carico i vettori
        xvect.append(xnew)

        # criterio di arresto e aggiorno
        if abs(xnew - xold) < toll:
            break
        else:
            xold = xnew

    return np.array(xvect)

#
# Lab 3: metodi bksub and fwsub
#
def bksub(A,b):
    """
    Algoritmo di sostituzione all'indietro - backward substitution
    Input:
    A: matrice quadrata triangolare superiore
    b: termine noto
    Output:
    x: soluzione del sistema lineare = b
    """

    # dimensione vettore b
    n = b.shape[0]

    # Verifichiamo che la matrice sia quadrata
    if A.shape[0] != A.shape[1]:
        raise RuntimeError("ERRORE: matrice non quadrata")

    # Verifichiamo che la matrice sia triangolare inferiore
    if (A != np.triu(A)).any():
        raise RuntimeError("ERRORE: matrice non triangolare superiore")

    # Verifichiamo che la matrice sia invertibile
    # Essendo triangolare, i suoi autovalori si trovano sulla diagonale principale
    if np.prod(np.diag(A)) == 0:
        raise RuntimeError("ERRORE: matrice singolare")

    # inizializzo il vettore x
    x = np.zeros(n)

    # x[n-1] = b[n-1]/A[n-1,n-1]
    x[-1] = b[-1] / A[-1, -1]

    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - A[i, i + 1 : n] @ x[i + 1 : n]) / A[i, i]

    # Versione alternativa: doppio ciclo for
    #  x = np.zeros(n)
    #  x[-1] = b[-1] / A[-1,-1]
    #
    #  for i in range(n-2,-1,-1):
    #    s = 0
    #
    #    for j in range(i,n):
    #      s = s + A[i,j] * x[j]
    #
    #    x[i] = (b[i] - s) / A[i,i]

    return x


def fwsub(A,b):
    """
    Algoritmo di sostituzione in avanti - forward substitution
    Input:
    A: matrice quadrata triangolare inferiore
    b: termine noto
    Output:
    x: soluzione del sistema lineare = b
    """
  
    # dimesinoe termine noto b
    n = b.shape[0]

    # Verifichiamo che la matrice sia quadrata
    if A.shape[0] != A.shape[1]:
        raise RuntimeError("ERRORE: matrice non quadrata")

    # Verifichiamo che la matrice sia triangolare inferiore
    if (A != np.tril(A)).any():
        raise RuntimeError("ERRORE: matrice non triangolare inferiore")

    # Verifichiamo che la matrice sia invertibile
    # Essendo triangolare, i suoi autovalori si trovano sulla diagonale principale
    if np.prod(np.diag(A)) == 0:
        raise RuntimeError("ERRORE: matrice singolare")

    # inizializzo il vettore
    x = np.zeros(n)
    # costruzione forward substitution
    x[0] = b[0] / A[0, 0]

    for i in range(1, n):
        x[i] = (b[i] - A[i, 0:i] @ x[0:i]) / A[i, i]

    # Versione alternativa: doppio ciclo for
    #  x = np.zeros(n)
    #  x[0] = b[0] / A[0,0]
    #
    #  for i in range(1,n):
    #    s = 0
    #
    #    for j in range(0,i):
    #      s = s + A[i,j] * x[j]
    #
    #    x[i] = (b[i] - s) / A[i,i]

    return x