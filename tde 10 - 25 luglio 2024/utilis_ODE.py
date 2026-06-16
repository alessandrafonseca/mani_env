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
    N = np.atleast_1d(t_h).size
    d = np.atleast_1d(y0).size
    u_h = np.zeros((N, d))

    # ciclo iterativo che calcola i passi di Eulero esplicito
    u_h[0, :] = y0

    for it in range(N - 1):
        u_old = u_h[it, :]
        u_h[it + 1, :] = u_old + h * f(t_h[it], u_old)

    return t_h, u_h


def eulero_indietro(f, t0, t_max, y0, h):
    """Metodo di Eulero all'indietro
    Input:
      f     (lambda function)   Termine di destra dell'ODE, passata come
                                funzione di tempo e spazio, f = f(t, y)
      t0    (float)             Tempo iniziale
      t_max (float)             Tempo finale
      y0    (float)             Dato iniziale
      h     (float)             Passo temporale

    Output:
      t   (numpy.ndarray)     Griglia temporale
      u   (numpy.ndarray)     Approssimazioni della soluzione nei nodi temporali t_i
    """
    # parametri per il punto fisso
    nmax_pf = 300
    toll_pf = 1e-5

    # vettore dei nodi temporali
    t_h = np.linspace(t0, t_max, int((t_max - t0) / h) + 1)

    # inizializzazione del vettore soluzione
    N = np.atleast_1d(t_h).size
    d = np.atleast_1d(y0).size
    u_h = np.zeros((N, d))

    # ciclo iterativo che calcola i passi di Eulero esplicito
    u_h[0, :] = y0

    for it in range(N - 1):
        # definisco la lambda function phi per il metodo del punto fisso
        phi = lambda z: u_h[it] + h * f(t_h[it + 1], z)

        # chiamo il metodo del punto fisso
        u_h[it + 1] = puntofisso(phi, u_h[it, :], nmax_pf, toll_pf)[-1]

    return t_h, u_h


def puntofisso(phi, x0, nmax=100, toll=1.0e-6):
    """Metodo punto fisso
    Input:
      𝚙𝚑𝚒   (lambda function)   Funzione di iterazione
      𝑥0    (float)             Punto di partenza
      𝚗𝚖𝚊𝚡  (float)             Numero massimo di iterazione
      𝚝𝚘𝚕𝚕  (float)             Tolleranza richiesta

    Output:
      𝚡𝚟𝚎𝚌𝚝 (numpy.ndarray)     Vettore delle iterate.
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
