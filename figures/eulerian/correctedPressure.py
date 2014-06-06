# Before the time-stepping:

# Formulate the pressure correction problem
a2 = inner(grad(q), grad(p))*dx		# ${\langle}{\nabla}{q},{\nabla}{p^n}{\rangle}$
L2 = inner(grad(q), grad(p0))*dx\	# ${\langle}{\nabla} q, \nabla p^{n-1} \rangle - \langle \nabla \cdot \mathbf{u}^{\star}, q\rangle/\Delta t_n$ 
	  - (1/k)*q*div(u1)*dx

# Pre-assemble the LHS
A2 = assemble(a2)

...

# During the time-stepping:

# Assemble the RHS
b = assemble(L2)

# Apply the Dirichlet velocity boundary condition b.c
if len(bcPressure) == 0: normalize(b)
[bc.apply(A2, b) for bc in bcPressure]

# Solve for the corrected pressure
solve(A2, p1.vector(), b)
if len(bcPressure) == 0: normalize(p1.vector())

