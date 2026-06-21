import numpy as np


def eulero_avanti(f, t0, t_max, y0, h):
    """Risolve il problema di Cauchy

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
    t_h = np.linspace(t0, t_max, int((t_max - t0) / h) + 1)

    # inizializzazione del vettore soluzione
    N = len(t_h)
    d = len(y0)
    u_h = np.zeros((N, d))

    # ciclo iterativo che calcola i passi di Eulero esplicito
    u_h[0, :] = y0

    for it in range(N - 1):
        u_old = u_h[it, :]
        u_h[it + 1, :] = u_old + h * f(t_h[it], u_old)

    return t_h, u_h


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
