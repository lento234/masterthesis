# Before the time-stepping:

# Define: $\mathbf{u}^{n-1/2} = (\mathbf{u}^{\star}+\mathbf{u}^{n-1})/2$
U  = 0.5*(u0 + u)

# Formulate the tentative velocity problem
F1 = (1/k)*inner(v, u - u0)*dx \
            + inner(v, grad(u0)*u0)*dx \
            + inner(epsilon(v), sigma(U, p0, nu))*dx \
            + inner(v, p0*n)*ds \
            - beta*nu*inner(grad(U).T*n,v)*ds \
            - inner(v, f)*dx

# Extract the LHS, and the RHS
a1 = lhs(F1)
L1 = rhs(F1)

# Pre-assemble the LHS
A1 = assemble(a1)

...

# During the time-stepping:

# Assemble the RHS
b = assemble(L1)

# Apply the Dirichlet velocity boundary condition b.c
[bc.apply(A1, b) for bc in bcVelocity]

# Solve for the Tentative velocity
solve(A1, u1.vector(), b, "gmres", "default")
