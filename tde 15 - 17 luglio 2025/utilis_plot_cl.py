import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.colors import Colormap

def xtplot(x, t, u, plot_type='animation', color='blue', piecewise=True, name_gif='solution_animation'):
    """Plot della soluzione:
    Input:
    x         (numpy.ndarray)-> vector  Baricentri delle celle
    t         (numpy.ndarray)-> vector  Tempi d'evoluzione
    u         (numpy.ndarray)-> matrix  Approssimazione della soluzione. Vige la convenzione uij = u(xi,tj).
    type      (string)                  'animation' o 'surface'
    color     (string)                  Colore
    piecewise (bool)                    Costante a tratti True or False
    name_gif  (string)                  Nome gif
    """
    if plot_type == 'animation':
        animate_conservation_laws(x, t, u, color, piecewise, name_gif)
    elif plot_type == 'surface':
        surface_conservation_laws(x, t, u)
    else:
        raise ValueError("Errore: il plot_type in input è sbagliato")

def plot_frame(x, t, u, i, color='blue', piecewise=True):
    """Plotta il frame i-esimo.    
    Input:
    x         (numpy.ndarray)-> vector  Baricentri delle celle
    t         (numpy.ndarray)-> vector  Tempi d'evoluzione
    u         (numpy.ndarray)-> matrix  Approssimazione della soluzione. Vige la convenzione uij = u(xi,tj).
    i         (int)                     Frame
    color     (string)                  Colore
    piecewise (bool)                    Costante a tratti True or False
    """
    if i >= len(t):
        raise ValueError("Errore: indice frame fuori intervallo")
    
    fig, ax = plt.subplots(figsize=(6, 5))
    fvplot(x, u[:, i], color, piecewise)
    ax.set(xlabel='x', ylabel='u(x,t)',
           title=f'Current time: t = {t[i]:.2f}',
           xlim=(x[0] - (x[1] - x[0]), x[-1] + (x[1] - x[0])),
           ylim=(np.min(u) - 0.3 * (np.max(u) - np.min(u)), 
                 np.max(u) + 0.3 * (np.max(u) - np.min(u))))

# funzione utile per plottare
def fvplot(x, v, c, pcw):
  if (pcw):
    h = (x[1]-x[0])/2
    plt.plot([x[0]-h, x[0]+h], [v[0], v[0]], color=c)

    for i in range(1,len(x)):
      plt.plot([x[i]-h, x[i]-h], [v[i-1], v[i]],'--', color=c)
      plt.plot([x[i]-h, x[i]+h], [v[i], v[i]], color=c)

  else:
      plt.plot(x,v,color=c)

def save_gif(draw_frame, frames, filename, dt=1.0/24.0):
    """Salva l'animazione in formato GIF."""
    images = []
    for i in range(frames):
        draw_frame(i)
        fig = plt.gcf()
        fig.canvas.draw()
        images.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close(fig)
    # imageio.mimsave(f'{filename}.gif', images, duration=dt)
    imageio.mimsave(filename.replace(".gif", "") + ".gif", images, duration = dt)

def animate_conservation_laws(x, t, u, color='blue', piecewise=True, name = 'solution_animation'):
    """Crea un'animazione della soluzione."""
    draw_frame = lambda i: plot_frame(x, t, u, i, color, piecewise)
    save_gif(draw_frame, len(t), name)
    from IPython.display import Image, display
    name_gif = name+'.gif'
    display(Image(name_gif))

def surface_conservation_laws(x, t, u):
    """Plotta la soluzione come superficie 3D."""
    X, T = np.meshgrid(x, t, indexing='ij')
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # Prima figura: 3D surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, T, u, cmap='plasma', edgecolor='k', linewidth=0.3, alpha=0.9, antialiased=True, shade=True)
    ax1.set(xlabel='x', ylabel='t', zlabel='u(x,t)', title='Surface plot')
    ax1.view_init(elev=30, azim=-90) 

    # Limiti opzionali
    epsilon = 1.e-1
    ax1.set_xlim(X.min()-epsilon, X.max()+epsilon)
    ax1.set_ylim(T.min()-epsilon, T.max()+epsilon)
    ax1.set_zlim(u.min()-epsilon, u.max()+epsilon)

    # seconda figura 
    ax2 = fig.add_subplot(1, 2, 2)
    pcolor = ax2.pcolor(X, T, u, cmap='plasma')
    fig.colorbar(pcolor, ax=ax2)
    ax2.set(xlabel='x', ylabel='t', title='Top view')
    ax2.set_aspect('equal')

    plt.show()