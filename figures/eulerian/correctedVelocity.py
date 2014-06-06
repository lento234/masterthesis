# Before the time-stepping:

# Formulate the velocity correction problem
a3 = inner(v, u)*dx	# ${\langle}{\mathbf{u}^n,\mathbf{v}}$
L3 = inner(v, u1)*dx - k*inner(v, grad(p1 - p0))*dx # ${\langle}{\mathbf{u}^{\star},\mathbf{v}} - {\Delta}t_n{\langle}{\nabla}(p^n-{p^{n-1}}),\mathbf{v}{\rangle}$

# Pre-assemble the LHS
A3 = assemble(a3)

...

# During the time-stepping:

# Assemble the RHS
b = assemble(L3)

# Apply the Dirichlet velocity boundary condition b.c
[bc.apply(A3, b) for bc in bcVelocity]

# Solve for the corrected pressure
solve(A3, u1.vector(), b, "gmres", 'default')
