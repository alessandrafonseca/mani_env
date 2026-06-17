import numpy as np

def Jacobi_Bc(A, b = None):

  D, E, F = DEFsplit(A)
  M = D
  N = E+F

  Minv = np.diag(1.0/np.diag(M))
  B = Minv @ N

  if(b is None):
    return B
  else:
    c = Minv @ b
    return B, c
    
from scipy.linalg import solve_triangular

def GS_Bc(A, b = None):

  D, E, F = DEFsplit(A)
  M = D-E
  N = F

  B = solve_triangular(M, N, lower = True)

  if(b is None):
    return B
  else:
    c = solve_triangular(M, b, lower = True)
    return B, c
    
def DEFsplit(A):
  D = np.diag(np.diag(A))
  E = -np.tril(A, k = -1)
  F = -np.triu(A, k = 1)
  return D, E, F
  
def iterative_solve(A, b, x0, method, nmax, rtoll):

  r = A @ x0 - b
  bnorm = np.linalg.norm(b)

  if(method == 'Jacobi'):
    B, c = Jacobi_Bc(A, b)
  elif(method == 'GS'):
    B, c = GS_Bc(A, b)
  else:
    raise RuntimeError("Metodo sconosciuto.")

  k = 0
  xiter = [x0]

  while( (np.linalg.norm(r)/bnorm) > rtoll  and k < nmax):
    xold = xiter[-1]
    xnew = B @ xold + c
    xiter.append(xnew)
    r = A @ xnew - b
    k = k+1

  return xiter
  
def eulero_avanti(f, t0, t_max, y0, h):
    """ Risolve il problema di Cauchy
    
     y'   = f(t,y)
     y(0) = y0
    
     utilizzando il metodo di Eulero in avanti (esplicito):
     u^(n+1) = u^n + h*f^n
    
     L'equazione differenziale ordinaria può essere in generale vettoriale
     (y=f(t,y) in R^d)
     per d=1 si ottiene il caso scalare.
    
     Input:
           f: lambda function che descrive il problema di Cauchy.
               Riceve in input due argomenti: f=f(t,y), con y vettore di lunghezza d
           t0, t_max: estremi dell'intervallo temporale di soluzione
           y0: dato iniziale del problema di Cauchy (vettore di lunghezza d)
           h: ampiezza de passo di discretizzazione temporale
     ATTENZIONE: controllare che l'output di f e il dato y0 siano vettori della stessa lunghezza!
    
     Output:
           t_h = vettore degli istanti in cui viene calcolata la soluzione discreta (lunghezza N)
           u_h = soluzione discreta calcolata nei nodi temporali t_h (matrice di dimensioni N x d)
    """
    
    # vettore dei nodi temporali
    t_h = np.linspace(t0, t_max, int((t_max-t0)/h)+1)

    # inizializzazione del vettore soluzione
    N = len(t_h)
    d = len(y0)
    u_h = np.zeros((N, d))

    # ciclo iterativo che calcola i passi di Eulero esplicito
    u_h[0, :] = y0

    for it in range(N-1):
        u_old = u_h[it, :]
        u_h[it+1, :] = u_old + h*f(t_h[it], u_old)
    
    return t_h, u_h

