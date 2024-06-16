"""
@Problem: 2D hyperelastic Cook's membrane. A benchmark example.

@Formulation: Mixed displacement-pressure formulation.

@Material: Incompressible Neo-Hookean.

@author: Dr Chennakesava Kadapa

Created on Sun 16-Jun-2024
"""


from fenics import *
from dolfinx import fem, io


# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 3



mesh = Mesh("Cooksmembrane2d-nelem8-P1.xml")

cell_tags  = MeshFunction('size_t', mesh, "Cooksmembrane2d-nelem8-P1_physical_region.xml")
facet_tags = MeshFunction('size_t', mesh, "Cooksmembrane2d-nelem8-P1_facet_region.xml")


dx = Measure("dx", domain=mesh, subdomain_data=cell_tags)
ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)



# FE Elements
# Quadratic element for displacement
U2 = VectorElement("CG", mesh.ufl_cell(), 2)
# Linear element for pressure
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)


# DOFs
TH = MixedElement([U2, P1])
# Total space for all DOFs
ME = FunctionSpace(mesh, TH)


# Define test functions in weak form
dw = TrialFunction(ME)
# Test function
(u_test, p_test)  = TestFunctions(ME)

# Define actual functions with the required DOFs
w = Function(ME)
# current DOFs
(u, p) = split(w)

# DOFs at previous load step
w_old = Function(ME)
(u_old, p_old) = split(w_old)


# Boundary conditions

# Left edge is fixed
bc1 = DirichletBC(ME.sub(0).sub(0),0.0,facet_tags, 1) # X-dof
bc2 = DirichletBC(ME.sub(0).sub(1),0.0,facet_tags, 1) # Y-dof


bcs = [bc1, bc2]


# Parameter values
E = 240.565  # MPa
nu = 0.4999 # Poisson's ratio
G = E/(2*(1 + nu))
K = E/(3*(1 - 2*nu))

mu = Constant(G)
lmbda = Constant(E*nu/(1+nu)/(1-2*nu))
kappa = Constant(K)

# traction
traction = Expression(("0.0","6.25*t"), t=0, degree=1)

# body force
f = Constant((0,0))

# When you create Function it gets zero intially. Therefore there is no need code below

d = len(u)
I = Identity(d)
F = variable(I + grad(u))
J = det(F)

C = F.T*F
# Free Energy Function
Psi = mu/2*(J**(-2/3)*tr(C) - mesh.topology().dim()) + p*(J-1-p/2/kappa)

#PK1 = diff(Psi, F) + p*J*inv(F.T)
PK1 = J**(-2/3)*G*(F - 1/3*tr(C)*inv(F.T)) + J*p*inv(F.T)


# Weak form
L = inner(PK1, grad(u_test))*dx + inner((J-1 - p/kappa), p_test)*dx - dot(traction, u_test)*ds(2)
dL = derivative(L, w, dw)


CoupledProblem = NonlinearVariationalProblem(L, w, bcs=bcs, J=dL)

# Set up the non-linear solver
solver  = NonlinearVariationalSolver(CoupledProblem)

# Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'mumps'  
prm['newton_solver']['absolute_tolerance'] = 1.E-6
prm['newton_solver']['relative_tolerance'] = 1.E-6
prm['newton_solver']['maximum_iterations'] = 30
prm['newton_solver']['convergence_criterion'] = 'incremental'


num_steps = 5
dt = 0.2
# Time-stepping
t = 0


# Output file setup
file_results = XDMFFile("Cooksmembrane2D-nelem8.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Give fields descriptive names
u_v = w.sub(0)
u_v.rename("displacement","")

p_v = w.sub(1)
p_v.rename("p", "")


# function to write results to XDMF at time t
def writeResults(t):

    # Displacement, pressure penalty term
    file_results.write(u_v,t)
    file_results.write(p_v,t)

writeResults(0)

for timeStep in range(num_steps):

    # Update current time
    t += dt
    traction.t = t

    # Solve the problem
    # Compute solution
    (iter, converged) = solver.solve()

    writeResults(t)


