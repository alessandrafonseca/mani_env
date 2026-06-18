import numpy as np


def fwsub(A, b):
    """
    Algoritmo di sostituzione in avanti - forward substitution.

    Input:
      A   (numpy.ndarray)   Matrice quadrata triangolare inferiore
      b   (numpy.ndarray)   Termine noto

    Output:
      x soluzione del sistema Ax = b, restituita come vettore numpy.
    """

    # dimesione termine noto b
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


def bksub(A, b):
    """
    Algoritmo di sostituzione all'indietro - backward substitution.

    Input:
      A   (numpy.ndarray)   Matrice quadrata triangolare superiore
      b   (numpy.ndarray)   Termine noto

    Output:
      x soluzione del sistema Ax = b, restituita come vettore numpy.
    """

    # inizializzo il vettore x
    x = []
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

def pmedcomp(f, a, b, N):
  """ Formula del punto medio composita
  Input:
     f:   funzione da integrare
     a:   estremo inferiore intervallo di integrazione
     b:   estremo superiore intervallo di integrazione
     N:   numero di sottointervalli (N = 1 formula di integrazione semplice)
  Output:
     I:   integrale approssimato """

  h = (b-a)/N                 # ampiezza sottointervalli
  x = np.linspace(a, b, N+1)  # griglia spaziale
  xL, xR = x[:-1], x[1:]      # liste dei nodi "sinistri" e "destri"
  xM = 0.5*(xL + xR)          # punti medi
  I = h*f(xM).sum()           # integrale approssimato

  return I
  
def trapcomp(f, a, b, N):
  """ Formula dei trapezi composita
  Input:
     f:   funzione da integrare
     a:   estremo inferiore intervallo di integrazione
     b:   estremo superiore intervallo di integrazione
     N:   numero di sottointervalli (N = 1 formula di integrazione semplice)
  Output:
     I:   integrale approssimato """

  h = (b-a)/N                           # ampiezza sottointervalli
  x = np.linspace(a, b, N+1)            # griglia spaziale
  xL, xR = x[:-1], x[1:]                # liste dei nodi "sinistri" e "destri"
  xM = 0.5*(xL + xR)                    # punti medi

  I = 0.5*h*(f(xL)+f(xR)).sum()         # integrale approssimato

  return I


def fvsolve(u0, f, df, L, T, h, dt, method):
    """Risolve un dato problema di trasporto utilizzando il metodo ai volumi finiti 1D.

    Input:
     u0            (lambda function)        Dato al tempo t = 0 (profilo iniziale)
     f             (lambda function)        Flusso dell'equazione,  f = f(u)
     df            (lambda function)        Derivata del flusso, df = f'(u)
     L             (float)                  Lunghezza dell'intervallo spaziale
     T             (float)                  Tempo finale
     h             (float)                  Grandezza delle celle
     dt            (float)                  Passo temporale
     method        (string)                 Metodo da utilizzare per i flussi

    Output:
    xc     (numpy.ndarray)-> vector  Baricentri delle celle
    t      (numpy.ndarray)-> vector  Tempi d'evoluzione
    u      (numpy.ndarray)-> matrix  Approssimazione della soluzione. Vige la convenzione uij = u(xi,tj).
    """

    # costruzione griglie spaziali e temporali
    ncells = int(np.ceil(L / h))  # numero celle
    nt = int(np.ceil(T / dt) + 1)  # numero nodi temporali
    x = np.linspace(0, L, ncells + 1)  # griglia spaziale
    xL = x[0:-1]  # nodi sinistri
    xR = x[1:]  # nodi destri
    xc = (xL + xR) / 2.0  # centri delle celle

    t = np.linspace(0, T, nt)  # nodi temporali

    # Inizializzazione soluzione
    u = np.zeros((ncells, nt))
    u[:, 0] = u0(xc)

    # Ciclo temporale
    for n in range(nt - 1):
        # Soluzione estesa
        uex = np.append([u0(x[0])], u[:, n])
        uex = np.append(uex, u[-1, n])

        # Calcolo del flusso
        if method == "UPWIND":
            flusso1 = upwind_flux(f, df, uex[0:-2], uex[1:-1])
            flusso2 = upwind_flux(f, df, uex[1:-1], uex[2:])
        elif method == "GODUNOV":
            flusso1 = godunov_flux(f, df, uex[0:-2], uex[1:-1])
            flusso2 = godunov_flux(f, df, uex[1:-1], uex[2:])

        # Passo temporale
        u[:, n + 1] = u[:, n] + (dt / h) * (flusso1 - flusso2)

    return xc, t, u


# Implementazione del flusso "alla upwind"
def upwind_flux(f, df, uL, uR):
    if np.min(df(uL)) >= 0 and np.min(df(uR)) >= 0:
        F = f(uL)
    else:
        F = f(uR)

    return F


# Implementazione del flusso "alla Godunov"
def godunov_flux(f, df, uL, uR):
    iL = np.minimum(np.array(uL), np.array(uR))
    iR = np.maximum(np.array(uL), np.array(uR))
    g = np.linspace(0, 1, 1000).reshape(1, 1000)

    iL = iL.reshape(len(iL), 1)
    iR = iR.reshape(len(iR), 1)
    g = g.reshape(1, 1000)

    itot = f(iL @ g + iR @ (1 - g))

    imins = itot.min(axis=1)
    imaxs = itot.max(axis=1)

    candidates = imins
    d = np.sign(uR - uL)
    candidates[d < 0] = imaxs[d < 0]
    return np.array(candidates)
