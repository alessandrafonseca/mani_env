# import
import numpy as np


## Ricerca degli zeri: bisezione - Newton - punto fisso
# definizione del metodo di bisezione
def bisez(f, a, b, toll=1e-6):
    # controllo se gli estremi sono una bracket (opzionale)
    if f(a) * f(b) >= 0:
        raise RuntimeError("ERRORE: lintervallo [a,b] non è una bracket")

    # inizializzazione
    xvect = []

    while abs(b - a) > toll:
        x = 0.5 * (a + b)

        # primo controllo se x è uno zero (opzionale)
        if f(x) == 0:
            xvect.append(x)
            print("x è esattamente uno zero della funzione")
            break

        # metodo di bisezione
        if f(a) * f(x) > 0:
            a = x
        else:
            b = x

        xvect.append(x)

    return np.array(xvect)


# Definizione del metodo di Newton
def newton(f, df, x0, nmax=100, toll=1e-6, m=1):
    xvect = []
    xold = x0

    for nit in range(nmax):
        # verifica che la derivata prima non è nulla
        if df(xold) == 0:
            raise RuntimeError("ERRORE: Derivata prima nulla \n")
        else:
            # calcolo il nuovo punto
            xnew = xold - m * f(xold) / df(xold)
            # carico i vettori
            xvect.append(xnew)

        # criterio di arresto e aggiorno
        if abs(xnew - xold) < toll:
            break
        else:
            xold = xnew

    return np.array(xvect)


def puntofisso(phi, x0, nmax=100, toll=1.0e-6):
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
