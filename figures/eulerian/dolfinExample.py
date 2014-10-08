from dolfin import *
		
# Generate unit square mesh: $24\times{24}$
mesh = UnitSquareMesh(24, 24)
		
# Define Function space: 1st order, Continuous-Galerkin
V = FunctionSpace(mesh,"CG",1)
		
# Define Dirichlet boundary conditions expression
# $u_0 = \sin{x}\cdot\cos{y}$
u0 = Expression("sin(10*x[0])*cos(10*x[1])")

# Function that defines the boundary points
def u0_boundary(x, on_boundary):
    return on_boundary

# Define the boundary condition
# $u(x) = u_0(x), x \mathrm{on} \partial{\Omega}$
bc = DirichletBC(V, u0, u0_boundary)			
				
# Define the variational problem
u = TrialFunction(V)  # Trial functions
v = TestFunction(V)   # Test functions

# $f=100{\cdot}\sin(x){\cdot}\cos(y)$
f = Expression('100*sin(10*x[0])*cos(10*x[1])')

# LHS: $a=-\int{\nabla}{u}{\nabla}{v} \mathrm{d}x$
a = -inner(nabla_grad(u), nabla_grad(v))*dx 

# RHS:  $L=\int{fv} \mathrm{d}x$
L = f*v*dx

# Solve the Poisson problem
u = Function(V)       # Define the solution	
solve(a == L, u, bc)  # $a(u,v) = L(v)$

# Plot the result
plot(u, interactive=True)
