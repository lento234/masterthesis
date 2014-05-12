from dolfin import *
		
# Generate unit square mesh: $24\times{24}$
mesh = UnitSquareMesh(24, 24)
		
# Define Function space: 1st order, Continuous-Galerkin
V = FunctionSpace(mesh,"CG",1)
		
# Define boundary conditions
# $u_0 = \sin{x}\cdot\cos{y}$
u0 = Expression("sin(x[0])*cos(x[1])")

def u0_boundary(x, on_boundary):
    return on_boundary

# Define the boundary condition
# $u(x) = u_0(x), x \mathrm{on} \partial{\Omega}$
bc = DirichletBC(V, u0, u0_boundary)			
				
# Define the variational problem
u = TrialFunction(V)  # Trial function
v = TestFunction(V)   # Test function
f = Constant(2.)      # $f=2$
a = -inner(nabla_grad(u), nabla_grad(v))*dx # LHS: $a = -\int{\nabla}{u}{\nabla}{v} \mathrm{d}x$
L = f*v*dx            # RHS:  $L = \int{fv} \mathrm{d}x$

# Solve the Poisson problem
u = Function(V)       # Define the solution	
solve(a == L, u, bc)  # $a(u,v) = L(v)$
		
# Plot the result
plot(u)
