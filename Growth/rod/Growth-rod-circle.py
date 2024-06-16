"""
Growth-induced deformation of rods.

Example from Kadapa et al. JMPS, 148:104289, 2021.
https://www.sciencedirect.com/science/article/pii/S0022509620304919

The code is for simulating the circle shape in Figure 22b.

@Formulation: Mixed displacement-pressure formulation.

@author: Dr Chennakesava Kadapa

Created on Sun 16-Jun-2024
"""


from fenics import *
import numpy as np


# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 4


#read mesh
mesh = Mesh("rod3d-P1-nelem50.xml")

cell_tags  = MeshFunction('size_t', mesh, "rod3d-P1-nelem50_physical_region.xml")
facet_tags = MeshFunction('size_t', mesh, "rod3d-P1-nelem50_facet_region.xml")


# Define a ds measure for each face, necessary for applying traction BCs.
dx = Measure('dx', domain=mesh, subdomain_data=cell_tags)
ds = Measure('ds', domain=mesh, subdomain_data=facets)


# Parameter values
E = 1e6  # 20 MPa
nu = 0.4999 # Poisson's ratio
G = E/(2*(1 + nu))
K = E/(3*(1 - 2*nu))

mu = Constant(G)
lmbda = Constant(E*nu/(1+nu)/(1-2*nu))
kappa = Constant(K)



# FE Elements
# Quadratic element for displacement
U2 = VectorElement("CG", mesh.ufl_cell(), 2)
# Linear element for pressure
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)


# DOFs
TH = MixedElement([U2, P1])
ME = FunctionSpace(mesh, TH) # Total space for all DOFs



# Define test functions in weak form
dw = TrialFunction(ME)                                   
(u_test, p_test)  = TestFunctions(ME)    # Test function

# Define actual functions with the required DOFs
w = Function(ME)
# current DOFs
(u, p) = split(w)

# DOFs at previous load step
w_old = Function(ME)
(u_old, p_old) = split(w_old)



# Boundary conditions

# Homogeneous deformation mechanical BCs
bcs_a  = DirichletBC(ME.sub(0).sub(0),0.0,facet_tags,2) # left face x-fix
bcs_b  = DirichletBC(ME.sub(0).sub(1),0.0,facet_tags,2) # left face y-fix
bcs_c  = DirichletBC(ME.sub(0).sub(2),0.0,facet_tags,2) # left face z-fix

bcs = [bcs_a, bcs_b, bcs_c]


# Define kinematic variables
d = len(u)
I = Identity(d)
F = variable(I + grad(u))
J = det(F)

t   = 0.0

Fg = Expression( (("1.0",0.0,0.0),\
                  (0.0,"1.0",0.0),\
                  (0.0,0.0,"1.0-0.5*3.1415*x[0]*t")),\
                 t=t, degree=2)

FgInv = inv(Fg)
Jg = det(Fg)
Fe = F*FgInv
Je = det(Fe)
Ce = Fe.T*Fe
# Free Energy Function
Psi = mu/2*(Je**(-2/3)*tr(Ce) - 3) + p*(Je-1-p/2/kappa)

#PK1 = diff(Psi, F) + p*J*inv(F.T)
PK1 = Je**(-2/3)*G*(Fe - 1/3*tr(Ce)*inv(Fe.T)) + Je*p*inv(Fe.T)


# Weak form
L = inner(PK1, grad(u_test))*Jg*dx + inner((Je-1 - p/kappa), p_test)*Jg*dx
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


num_steps = 100
dt = 1.0/num_steps
# Time-stepping
t = 0


# Output file setup
file_results = XDMFFile("rod3d-circle.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Give fields descriptive names
u_v = w.sub(0)
u_v.rename("displacement","")

p_v = w.sub(1)
p_v.rename("pressure", "")


# function to write results to XDMF at time t
def writeResults(t):

    # Displacement, pressure penalty term
    file_results.write(u_v,t)
    file_results.write(p_v,t)

writeResults(0)

for timeStep in range(num_steps):
    # Update current time
    t += dt
    Fg.t = t

    print("\n\n Load step = ", timeStep+1)
    print("     Time      = ", t)

    # Solve the problem
    # Compute solution
    (iter, converged) = solver.solve()

    writeResults(t)



