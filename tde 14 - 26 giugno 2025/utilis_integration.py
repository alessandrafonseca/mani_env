import numpy as np

def pmedcomp(f, a, b, N):
  """
  Formula del punto medio composita
  Input:
     f:   funzione da integrare
     a:   estremo sinistro dell'intervallo di integrazione
     b:   estremo destro dell'intervallo di integrazione
     N:   numero di sottointervalli (N = 1 formula di integrazione semplice)
  Output:
     I:   integrale approssimato
  """

  h = (b-a)/N                 # ampiezza sottointervalli
  x = np.linspace(a, b, N+1)  # griglia spaziale
  xL, xR = x[:-1], x[1:]      # liste dei nodi "sinistri" e "destri"
  xM = 0.5*(xL + xR)          # punti medi

  I = h*f(xM).sum()           # integrale approssimato

  return I

def trapcomp(f, a, b, N):
  """
  Formula dei trapezi composita
  Input:
     f:   funzione da integrare
     a:   estremo sinistro dell'intervallo di integrazione
     b:   estremo destro dell'intervallo di integrazione
     N:   numero di sottointervalli (N = 1 formula di integrazione semplice)
  Output:
     I:   integrale approssimato
  """

  h = (b-a)/N                           # ampiezza sottointervalli
  x = np.linspace(a, b, N+1)            # griglia spaziale
  xL, xR = x[:-1], x[1:]                # liste dei nodi "sinistri" e "destri"

  I = 0.5*h*(f(xL)+f(xR)).sum()         # integrale approssimato

  return I

def simpcomp(f, a, b, N):
  """
  Formula di Cavalieri-Simpson composita
  Input:
     f:   funzione da integrare
     a:   estremo sinistro dell'intervallo di integrazione
     b:   estremo sinistro dell'intervallo di integrazione
     N:   numero di sottointervalli (N = 1 formula di integrazione semplice)
  Output:
     I:   integrale approssimato
  """

  h = (b-a)/N                                     # ampiezza sottointervalli
  x = np.linspace(a, b, N+1)                      # griglia spaziale
  xL, xR = x[:-1], x[1:]                          # liste dei nodi "sinistri" e "destri"
  xM = 0.5*(xL + xR)                              # punti medi

  I = (h/6.0)*(f(xL)+4*f(xM)+f(xR)).sum()         # integrale approssimato

  return I