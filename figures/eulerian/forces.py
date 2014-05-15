...

def epsilon(u):
    "Returns symmetric gradient"
    return 0.5*(grad(u) + grad(u).T)

def sigma(u,p,nu):
    "Returns stress tensor"
    return 2*nu*epsilon(u) - p*Identity(u.cell().d)

# Define the normal function
n = FacetNormal(mesh)

# Define the unit vectors
eX = Constant((1.0, 0.0))
eY = Constant((0.0, 1.0))

# Define the line integrator
ds = Measure("ds")[boundaryDomains]
noSlip = 2 # No-slip boundary identification = 2 

# Determine the forces
# Integrate the forces over the boundaryDomain == noSlip
L = assemble(inner(inner(sigma(u,p,nu), n), eY)*ds[noSlip]) # Lift
D = assemble(inner(inner(sigma(u,p,nu), n), eY)*ds[noSlip]) # Drag
