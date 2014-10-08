from dolfin import *

...

# Define Function spaces
# $X$ : scalar-valued vorticity function space $W$
X = FunctionSpace(mesh, 'CG', 1) # 1st order, Continuous-Galerkin

# Define the trial and test function
omega = TrialFunction(X) # vorticity $\omega{\in}X$
v     = TestFunction(X)  # test function $v{\in}\hat{X}$

...

# Define the variation problem for vorticity
a = inner(omega,v)*dx    # $ \langle{\omega,v}\rangle$
b = inner(curl(u),v)*dx  # $ \langle{\nabla{\times}u,v}\rangle$

# Pre-Assemble the LHS
A = assemble(a)

...

# During the time-stepping
omega = Function(X) # Define the function
B = assemble(b)     # Assemble b
solve(A, omega.vector(), B) # Solve for vorticity
