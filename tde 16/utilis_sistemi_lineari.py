# import
import numpy as np
# funzione per la risoluzione di sistemi triangolari
from scipy.linalg import solve_triangular

## Metodi diretti: sostituzione in avanti e indietro
def fwsub(A, b):
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

    return x

def bksub(A, b):
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

    return x

## Metodi iterativi: Gauss-Seidel - Jacobi - metodo del gradiente a parametro dinamico
# Gauss-Seidel
def gauss_seidel_bg(A, b=None):
  """
  Metodi di Gauss-Seidel
  Input:
  A: matrice quadrata
  b: termine noto (facoltativo)
  Output:
  B: matrice di iterazione di Gauss-Seidel
  g: vettore di shifting di Gauss-Seidel (se abbiamo come input b)
  """

  # Definisco il precondizionatore
  P = np.tril(A)
  # Calcolo la matrice di iterazione
  P_inv_A = solve_triangular(P, A, lower=True)
  B = np.eye(A.shape[0]) - P_inv_A

  if(b is None):
    return B
  else:
    g = solve_triangular(P, b, lower=True)
    return B, g
  
# Gauss-Seidel
def gauss_seidel_bg(A, b=None):
  """
  Metodi di Gauss-Seidel
  Input:
  A: matrice quadrata
  b: termine noto (facoltativo)
  Output:
  B: matrice di iterazione di Gauss-Seidel
  g: vettore di shifting di Gauss-Seidel (se abbiamo come input b)
  """

  # Definisco il precondizionatore
  P = np.tril(A)
  # Calcolo la matrice di iterazione
  P_inv_A = solve_triangular(P, A, lower=True)
  B = np.eye(A.shape[0]) - P_inv_A

  if(b is None):
    return B
  else:
    g = solve_triangular(P, b, lower=True)
    return B, g

# iterative_solve
def iterative_solve(A, b, x0, B, g, nmax, rtoll):
  """
  Metodo di risoluzione utilizzando i metodi iterativi

  Input:
  A: matrice quadrata
  b: termine noto
  x0: vettore di innesco
  B: matrice di iterazione
  g: vettore di shifting
  nmax: numero massimo di iterazioni
  rtoll: tolleranza relativa richiesta

  Output:
  xiter: lista contenente tutte le iterate
  """

  # norma di b
  bnorm = np.linalg.norm(b)

  # inizializzazione
  xold = x0
  xiter=[x0]
  n = 0
  err = 1 + rtoll

  while n < nmax and err > rtoll:
    # passo iterativo
    xnew = B @ xold + g
    # carico la lista xiter
    xiter.append(xnew)
    # residuo di xold
    r =  A @ xnew - b
    # test di arresto
    err = np.linalg.norm(r) / bnorm
    # aggiorno
    xold = xnew

  return xiter

import numpy as np


def gdescent(A, b, x0, nmax=1000, rtoll=1e-6):
    """
    Metodo del gradiente a parametro dinamico per sistemi lineari.

    Input:
     A      Matrice del sistema
     b      Termine noto (vettore)
     x0     Guess iniziale (vettore)
     nmax   Numero massimo di iterazioni
     toll   Tolleranza sul test d'arresto (sul residuo relativo)

    Output:
     xiter  Lista delle iterate

    """
    norm = np.linalg.norm

    bnorm = norm(b)

    r = b - A @ x0

    xiter = [x0]
    it = 0

    while (norm(r) / bnorm) > rtoll and it < nmax:
        xold = xiter[-1]

        z = r
        rho = np.dot(r, z)
        q = A @ z
        alpha = rho / np.dot(z, q)
        xnew = xold + alpha * z
        r = r - alpha * q

        xiter.append(xnew)
        it = it + 1

    return xiter

