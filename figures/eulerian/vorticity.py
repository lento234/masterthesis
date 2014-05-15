
# Define the trial and test function
omega = TrialFunction(W)
v = TestFunction(W)

# Define the variation problem for vorticity
a = inner(omega,v)*dx    # $ \langle{\omega,v}\rangle$
b = inner(curl(u),v)*dx  # $ \langle{\nabla{\times}u,v}\rangle$

# Pre-Assemble the LHS
A = assemble(a)

...

# During the time-stepping
omega = Function(W) # Define the function
B = assemble(b)     # Assemble b
solve(A, omega.vector(), B) # Solve for vorticity
