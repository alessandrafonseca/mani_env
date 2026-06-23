def heatSolve(D,f,u0,L,h,T,dt,theta):
  """"
  Input:
     D      (float)                  Coefficiente di diffusione (positivo).
     f      (lambda function)        Forzante. Si assume f = f(x,t).
     u0     (lambda function)        Condizione iniziale.
     L      (float)                  Lunghezza dell'intervallo spaziale.
     h      (float)                  Passo della griglia spaziale.
     T      (float)                  Tempo finale
     dt     (float)                  Passo temporale.
     theta  (float)                  Parametro del theta-metodo.

  Output:
    V                               spazio elementi finiti
    u     (numpy.ndarray)-> matrix  Matrice contenente la soluzione
                                    approssimata del problema. Uij
                                    approssima u(dof_i, tj): ogni colonna è un
                                    tempo fissato.
    t      (numpy.ndarray)-> vector Griglia temporale.
  """""
  # Costruisco il dominio
  domain = Line(0, L)
  # Costruisco la mesh
  mesh = generate_mesh(domain, stepsize = h)
  # Costruisco lo spazio FEM di grado 1
  Vh = FEspace(mesh, 1)

  # Costruisco la griglia temporale
  nt = np.ceil(T/dt)+1
  t = np.zeros(int(nt))

  # Inizializzo la soluzione
  u = np.zeros((dofs(Vh).size, int(nt)))

  # Definisco la condizione iniziale
  u0h = fun2dof(interpolate(u0,Vh)) # Qui utilizziamo fun2dof(interpolate(.)) perchè vogliamo che u sia di tipo np.array (si veda la documentazione delle due funzioni all'interno di fem.py per maggiori dettagli)
  u[:, 0] = u0h

  # Definisco le funzioni di base
  uh = trial_function(Vh)
  vh = test_function(Vh)

  # Matrice di massa
  def m(u, v):
    return u*v*dx
  # Assemblaggio matrice di massa
  M = assemble(m(uh, vh))

  # Matrice di diffusione
  def a(u,v):
    return deriv(u)*deriv(v)*dx
  # Assemblaggio matrice di diffusione
  A = D*assemble(a(uh, vh))

  # Ciclo temporale
  for n in range(int(nt)-1):
    # Costruzioni termini noti al tempo dt e dt+1
    t_old = n*dt
    t_new = (n+1)*dt

    fold = lambda x: f(x,t_old)
    fnew = lambda x: f(x,t_new)

    fold_h = interpolate(fold, Vh)
    fnew_h = interpolate(fnew, Vh)

    # Costruzione funzionali lineari
    def l(f, v):
      return f*v*dx

    Fold = assemble(l(fold_h, vh))
    Fnew = assemble(l(fnew_h, vh))

    # Applico condizioni di Dirichlet alle matrici del sistema
    def isLeftNode(x):
      return x < 1e-12

    def isRightNode(x):
      return x > L - 1e-12

    dbc1 = DirichletBC(isLeftNode,  0.0)
    dbc2 = DirichletBC(isRightNode, 0.0)

    A = applyBCs(A, Vh, dbc1, dbc2)
    M = applyBCs(M, Vh, dbc1, dbc2)
    Fold = applyBCs(Fold, Vh, dbc1, dbc2)
    Fnew = applyBCs(Fnew, Vh, dbc1, dbc2)

    # Costruzione del sistema lineare e sua risoluzione
    B = (M/dt+theta*A)
    b = (M/dt-(1-theta)*A)@u[:,n] + theta*Fnew + (1-theta)*Fold

    from scipy.sparse.linalg import spsolve

    u[:,n+1] = spsolve(B, b)
    t[n+1] = t_new

  return Vh, u, t


def parabolicSolve(a,b,f,u0,L,h,T,dt,theta):
  """"
  Input:
     a      (float)                  Coefficiente di diffusione (positivo).
     b      (float)                  Velocità di trasporto.
     f      (lambda function)        Forzante. Si assume f = f(x,t).
     u0     (lambda function)        Condizione iniziale.
     L      (float)                  Lunghezza dell'intervallo spaziale.
     h      (float)                  Passo della griglia spaziale.
     T      (float)                  Tempo finale
     dt     (float)                  Passo temporale.
     theta  (float)                  Parametro del theta-metodo.

  Output:
    V                               spazio elementi finiti
    u     (numpy.ndarray)-> matrix  Matrice contenente la soluzione
                                    approssimata del problema. Uij
                                    approssima u(dof_i, tj): ogni colonna è un
                                    tempo fissato.
    t      (numpy.ndarray)-> vector Griglia temporale.
  """""
  # Costruisco il dominio
  domain = Line(0, L)
  # Costruisco la mesh
  mesh = generate_mesh(domain, stepsize = h)
  # Costruisco lo spazio FEM di grado 1
  Vh = FEspace(mesh, 1)

  # Costruisco la griglia temporale
  nt = np.ceil(T/dt)+1
  t = np.zeros(int(nt))

  # Inizializzo la soluzione
  u = np.zeros((dofs(Vh).size, int(nt)))

  # Definisco la condizione iniziale
  u0h = fun2dof(interpolate(u0,Vh))
  u[:, 0] = u0h

  # Definisco le funzioni di base
  uh = trial_function(Vh)
  vh = test_function(Vh)

  # Matrice di massa
  def m(u, v):
    return u*v*dx
  # Assemblaggio matrice di massa
  M = assemble(m(uh, vh))

  # Matrice di diffusione
  def a_diff(u,v):
    return deriv(u)*deriv(v)*dx
  # Assemblaggio matrice di diffusione
  A_diff = assemble(a_diff(uh, vh))

  # Matrice di trasporto
  def a_trasp(u,v):
    return deriv(u)*v*dx
  # Assemblaggio matrice di trasporto
  A_trasp = assemble(a_trasp(uh, vh))

  A = a*A_diff + b*A_trasp

  # Ciclo temporale
  for n in range(int(nt)-1):
    # Costruzioni termini noti al tempo dt e dt+1
    t_old = n*dt
    t_new = (n+1)*dt

    fold = lambda x: f(x,t_old)
    fnew = lambda x: f(x,t_new)

    fold_h = interpolate(fold, Vh)
    fnew_h = interpolate(fnew, Vh)

    # Funzionali lineari
    def l(f, v):
      return f*v*dx

    Fold = assemble(l(fold_h, vh))
    Fnew = assemble(l(fnew_h, vh))

    # Condizioni al bordo omogenee di tipo dirichlet
    def isLeftNode(x):
      return x < 1e-12

    def isRightNode(x):
      return x > L - 1e-12

    dbc1 = DirichletBC(isLeftNode,  0.0)
    dbc2 = DirichletBC(isRightNode, 0.0)

    A = applyBCs(A, Vh, dbc1, dbc2)
    M = applyBCs(M, Vh, dbc1, dbc2)
    Fold = applyBCs(Fold, Vh, dbc1, dbc2)
    Fnew = applyBCs(Fnew, Vh, dbc1, dbc2)

    # Costruzione del sistema lineare e sua risoluzione
    B = (M/dt+theta*A)
    b = (M/dt-(1-theta)*A)@u[:,n] + theta*Fnew +(1-theta)*Fold

    from scipy.sparse.linalg import spsolve

    u[:,n+1] = spsolve(B, b)
    t[n+1] = t_new

  return Vh, u, t