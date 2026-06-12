"""
Modulo per la visualizzazione e l'animazione di un sistema molla-massa.

Fornisce funzioni per disegnare e animare il comportamento dinamico di un
sistema meccanico molla-massa in risposta a soluzioni numeriche di ODE.

Funzioni principali:
    - plotSpring: disegna il sistema molla-massa
    - drawSpringframe: crea un frame singolo con il tempo indicato
    - savegif: salva una sequenza di frame come GIF animata
    - animate: crea e visualizza l'animazione della soluzione
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio


def plotSpring(dx, springcolor='k', boxcolor='orange', springalpha=0.5, boxalpha=1.0):
    """
    Disegna il sistema molla-massa in una posizione specificata.
    
    Rappresenta graficamente un sistema composto da una molla fissa a sinistra
    e una massa (box) spostabile a destra. La posizione della massa è controllata
    dal parametro dx (spostamento dalla posizione di riposo).
    
    Input:
        dx (float): Spostamento dalla posizione iniziale della massa
        springcolor (str): Colore della molla (default: 'k' per nero)
        boxcolor (str): Colore della massa/box (default: 'orange')
        springalpha (float): Trasparenza della molla, 0-1 (default: 0.5)
        boxalpha (float): Trasparenza della massa, 0-1 (default: 1.0)
    
    Output:
        Nessuno (modifica la figura matplotlib corrente)
    """
    
    # Calcolo della distanza di riferimento
    dh = 20 * dx
    
    # Parametri della spirale della molla
    x = np.linspace(0, 15*np.pi, 81)
    y = 0.3 * np.sin(2*x)
    
    # Disegno della molla traslata
    plt.plot((x[-1]+dh)*x/x[-1], y, color=springcolor, alpha=springalpha)
    x = x + dh
    
    # Parametri della massa (box rettangolare)
    hbox = 4.5
    # Disegno del box (rettangolo che rappresenta la massa)
    plt.fill([x[-1], x[-1], x[-1]+hbox, x[-1]+hbox, x[-1], x[-1]],
             [0, -0.6, -0.6, 0.6, 0.6, 0], color=boxcolor, alpha=boxalpha)
    
    # Impostazione dell'area di disegno
    plt.axis([-1, 80, -1.01, 1.7])
    plt.axis("off")


def drawSpringframe(dx, t=0.0):
    """
    Crea un frame singolo che mostra il sistema molla-massa con tempo indicato.
    
    Questo frame è adatto per essere incluso in un'animazione. Mostra sia
    la posizione "fantasma" in trasparenza sia la posizione attuale della massa.
    
    Input:
        dx (float): Spostamento attuale dalla posizione di riposo
        t (float): Tempo attuale, visualizzato in alto a destra (default: 0.0)
    
    Output:
        Nessuno (crea e mostra una figura matplotlib)
    """
    
    # Creazione di una nuova figura con aspetto orizzontale sottile
    plt.figure(figsize=(6, 0.75))
    
    # Disegno della posizione "fantasma" (trasparente) come riferimento
    plotSpring(0.0, springcolor='r', boxcolor='r',
               springalpha=0.1, boxalpha=0.1)
    
    # Disegno della posizione attuale (colori opachi)
    plotSpring(dx)
    
    # Visualizzazione del tempo corrente
    plt.text(0, 1.5, 't = %.1f' % t, fontsize=8)
    
    # Marca il punto fisso della molla
    plt.plot(0, 0, '.k', alpha=0.5)


def savegif(drawframe, frames, name, dt=1.0/24.0):
    """
    Salva una sequenza di frame come file GIF animato.
    
    Questa funzione genera automaticamente i frame chiamando la funzione
    'drawframe' per ogni frame e li combina in un file GIF.
    
    Input:
        drawframe (callable): Funzione che disegna l'i-esimo frame.
                             Deve accettare un indice intero come argomento.
        frames (int): Numero totale di frame da generare
        name (str): Nome del file GIF output (senza estensione .gif)
        dt (float): Durata di ogni frame in secondi (default: 1.0/24.0 per 24 fps)
    
    Output:
        Nessuno (crea file GIF nel disco)
    
    Note:
        - I frame intermedi vengono generati come immagini RGBA dal renderer matplotlib
        - Il file viene salvato come '<name>.gif'
        - La memoria intermedia viene liberata dopo il salvataggio
    """
    
    # Lista per accumulare le immagini di ogni frame
    arrays = []
    
    # Generazione di tutti i frame
    for i in range(frames):
        # Disegna l'i-esimo frame
        drawframe(i)
        
        # Ottiene la figura corrente
        fig = plt.gcf()
        
        # Renderizza la figura in un array RGBA
        fig.canvas.draw()
        arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
        
        # Chiude la figura per liberare memoria
        plt.close(fig)

    # Salva gli array come GIF animata
    imageio.mimsave(name.replace(".gif", "") + ".gif", arrays, duration=dt)


def animate(t_h, x_h):
    """
    Crea e visualizza l'animazione della soluzione di un'ODE.
    
    Interpola linearmente la soluzione tra i nodi temporali e crea un'animazione
    GIF che viene visualizzata nel notebook Jupyter.
    
    Input:
        t_h (numpy.ndarray): Vettore dei tempi ai nodi di risoluzione
        x_h (numpy.ndarray): Vettore degli spostamenti ai nodi di risoluzione
                            (stessa lunghezza di t_h)
    
    Output:
        Nessuno (visualizza l'animazione in Jupyter e crea file temporaneo GIF)
    
    Note:
        - La funzione genera un'interpolazione con 10 frame per unità di tempo
        - Crea un file GIF temporaneo che viene eliminato dopo la visualizzazione
        - Utile per visualizzare soluzioni numeriche di ODE del tipo molla-massa
    """
    
    # Creazione di una griglia temporale densa per un'animazione fluida
    # 10 frame per ogni unità di tempo simulato
    t = np.linspace(t_h[0], t_h[-1], int(10*(t_h[-1]-t_h[0])+1))
    
    # Interpolazione dell'indice più vicino per ogni tempo nella griglia densa
    ind = np.abs(t_h.reshape(-1, 1) - t.reshape(1, -1)).argmin(axis=0)
    
    # Estrazione dei valori di spostamento interpolati
    x = x_h[ind]

    # Definizione della funzione che disegna il frame i-esimo
    def drawframe(i):
        return drawSpringframe(x[i], t[i])
    
    # Generazione di un numero casuale per il file temporaneo (evita conflitti)
    rnd = np.random.randint(50000)
    
    # Salva l'animazione come GIF temporanea
    savegif(drawframe, frames=len(t), name="temp%d-gif.gif" % rnd, dt=0.1)
    
    # Visualizza l'animazione nel notebook Jupyter
    from IPython.display import Image, display
    display(Image("temp%d-gif.gif" % rnd),
            metadata={'image/gif': {'loop': True}})
    
    # Elimina il file temporaneo
    from os import remove
    remove("temp%d-gif.gif" % rnd)
