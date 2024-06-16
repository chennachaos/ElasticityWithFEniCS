"""
@Problem: 3D cube with isotropic growth. Result is pure dilation of the cube.

Final growth tensor is: 
Fg =[[11,0,0],[0,11,0,],[0,0,11]]
resulting in the final deformation that is ten times the original size.

@Formulation: Mixed displacement-pressure formulation.

@Material: Incompressible Neo-Hookean.

@author: Dr Chennakesava Kadapa

Created on Sun 16-Jun-2024
"""

from fenics import *

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 4


# Dimensions of the beam
lenX = 1.0 #mm
lenY = 1.0 #mm
lenZ = 1.0 #mm

# Number of elements in each coordinate direction 
nelemX = 2
nelemY = 2
nelemZ = 2
  
# Define a uniformly spaced box mesh
#mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(lenX,lenY,lenZ), nelemX, nelemY, nelemZ)

mesh = UnitCubeMesh . create (nelemX, nelemY, nelemZ , CellType.Type.hexahedron )
mesh.coordinates()[:,0] = mesh.coordinates()[:,0]*lenX
mesh.coordinates()[:,1] = mesh.coordinates()[:,1]*lenY
mesh.coordinates()[:,2] = mesh.coordinates()[:,2]*lenZ

tol = 1e-12

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0.0, tol) and on_boundary
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],lenX, tol) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0, tol) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],lenY, tol) and on_boundary  
class Back(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2],0.0, tol) and on_boundary
class Front(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2],lenZ, tol) and on_boundary


# Mark boundary subdomians
facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
DomainBoundary().mark(facets, 1)  # First, mark all boundaries with common index

# Next mark sepcific boundaries
Left().mark(facets, 2)
Right().mark(facets,3)
Bottom().mark(facets, 4)
Top().mark(facets,5)
Back().mark(facets, 6)
Front().mark(facets,7)

# Define a ds measure for each face, necessary for applying traction BCs.
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
bcs_a  = DirichletBC(ME.sub(0).sub(0),0.0,facets,2) # left face x-fix
bcs_b  = DirichletBC(ME.sub(0).sub(1),0.0,facets,4) # bottom face y-fix
bcs_c  = DirichletBC(ME.sub(0).sub(2),0.0,facets,6) # back face z-fix

bcs = [bcs_a, bcs_b, bcs_c]


d = len(u)
I = Identity(d)
F = variable(I + grad(u))
J = det(F)

g   = 1.0
g11 = g
g22 = g
g33 = g
t   = 0.0

Fg = Expression( (("1.0+g11*t",0.0,0.0),(0.0,"1.0+g22*t",0.0),(0.0,0.0,"1.0+g33*t")), g11=g11, g22=g22, g33=g33, t=t, degree=1)
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


num_steps = 50
dt = 0.2
# Time-stepping
t = 0


# Output file setup
file_results = XDMFFile("cube.xdmf")
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

    # Solve the problem
    # Compute solution
    (iter, converged) = solver.solve()

    writeResults(t)



