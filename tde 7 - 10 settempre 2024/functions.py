import numpy as np
from scipy.linalg import solve_triangular


def iterative_solve(A, b, x0, method, nmax, rtoll):
    """
    Solve a linear system iteratively using the specified method.

    Parameters:
        A (numpy.ndarray): The coefficient matrix of the linear system.
        b (numpy.ndarray): The right-hand side vector of the linear system.
        x0 (numpy.ndarray): The initial guess for the solution vector.
        method (str): The iterative method to use (either "Jacobi" or "GS").
        nmax (int): The maximum number of iterations.
        rtoll (float): The relative tolerance for convergence.

    Returns:
        xiter (list): A list containing the iterates of the solution vector.
    """
    r = A @ x0 - b
    bnorm = np.linalg.norm(b)

    if method == "Jacobi":
        B, c = Jacobi_Bc(A, b)
    elif method == "GS":
        B, c = GS_Bc(A, b)
    else:
        raise RuntimeError("Metodo sconosciuto.")

    k = 0
    xiter = [x0]

    while (np.linalg.norm(r) / bnorm) > rtoll and k < nmax:
        xold = xiter[-1]
        xnew = B @ xold + c
        xiter.append(xnew)
        r = A @ xnew - b
        k = k + 1

    return xiter


def DEFsplit(A):
    """
    Splits a given matrix A into its diagonal matrix D, negative lower triangular matrix E, and
    negative upper triangular matrix F.

    Parameters:
        A (numpy.ndarray): The input matrix.

    Returns:
        tuple: A tuple containing the diagonal matrix D, negative lower triangular matrix E, and
        negative upper triangular matrix F.
    """
    D = np.diag(np.diag(A))
    E = -np.tril(A, k=-1)
    F = -np.triu(A, k=1)
    return D, E, F


def Jacobi_Bc(A, b=None):
    """
    Calculates the solution Jacobi iteration matrix and the right-hand side vector for the
    Jacobi method.

    Parameters:
        A (numpy.ndarray): The coefficient matrix.
        b (numpy.ndarray, optional): The right-hand side vector. Defaults to None.

    Returns:
        numpy.ndarray: If b is None, returns the Jacobi iteration matrix B.
        tuple: If b is not None, returns the Jacobi iteration matrix B and the transformed
               right-hand side vector c.
    """
    D, E, F = DEFsplit(A)
    M = D
    N = E + F

    M_inv = np.diag(1.0 / np.diag(M))
    B = M_inv @ N

    if b is None:
        return B
    else:
        c = M_inv @ b
        return B, c


def GS_Bc(A, b=None):
    """
    Calculates the solution Gauss-Seidel iteration matrix and the right-hand side vector for the
    Gauss-Seidel method.

    Parameters:
        A (numpy.ndarray): The coefficient matrix.
        b (numpy.ndarray, optional): The right-hand side vector. Defaults to None.

    Returns:
        numpy.ndarray: The solution vector if b is None.
        tuple: If b is not None, returns the Gauss-Seidel iteration matrix B and the
               transformed right-hand side vector c.
    """
    D, E, F = DEFsplit(A)
    M = D - E
    N = F

    B = solve_triangular(M, N, lower=True)

    if b is None:
        return B
    else:
        c = solve_triangular(M, b, lower=True)
        return B, c


def newton(f, df, x0, nmax=100, tol=1e-6, m=1):
    """
    Approximates the root of a function using Newton's method.

    Parameters:
        f (callable): The function for which to find the root.
        df (callable): The derivative of the function.
        x0 (numpy.ndarray): The initial guess for the root.
        nmax (int): The maximum number of iterations (default: 100).
        tol (float): The tolerance for convergence (default: 1e-6).
        m (int): The multiplicative factor (default: 1).

    Returns:
        xvect (numpy.ndarray): An array containing the sequence of approximations to the root.
    """

    x_vect = []
    x_old = x0
    for _ in np.arange(nmax):
        if df(x_old) == 0:
            raise RuntimeError("ERRORE: Derivata prima nulla \n")
        else:
            x_new = x_old - m * f(x_old) / df(x_old)
            x_vect.append(x_new)

        if abs(x_new - x_old) < tol:
            break
        else:
            x_old = x_new

    return np.array(x_vect)


def bisez(f, a, b, tol=1e-6):
    """
    Perform the bisection method to find the zeros of a function within a given interval.

    Parameters:
        f (callable): The function for which to find the zeros.
        a (float): The left endpoint of the interval.
        b (float): The right endpoint of the interval.
        tol (float, optional): The tolerance for the convergence of the method. Defaults to
            1e-6.

    Returns:
    x_vect (numpy.ndarray): An array containing the intermediate values of x during the
        bisection method.

    Raises:
        RuntimeError: If the interval [a,b] is not a bracket (i.e., f(a) * f(b) >= 0).
    """
    if f(a) * f(b) >= 0:
        raise RuntimeError("ERRORE: l" "intervallo [a,b] non è una bracket")

    x_vect = []
    while abs(b - a) > tol:
        x = 0.5 * (a + b)
        if np.isclose(f(x), 0):
            x_vect.append(x)
            print("x è uno zero della funzione")
            break

        if f(a) * f(x) > 0:
            a = x
        else:
            b = x
        x_vect.append(x)

    return np.array(x_vect)
