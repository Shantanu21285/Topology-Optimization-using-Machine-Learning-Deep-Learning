using TopOpt
using Makie, GeometryBasics, GLMakie,NPZ # For visualization

# Material properties
E = 1.0  # Young's modulus
v = 0.3  # Poisson's ratio

# Force applied
f = -1.0; # Downward force magnitude

# Number of elements in each dimension
nels = (30, 10, 10)  # Adjust based on desired resolution and computational resources

# Define the problem, assuming a cantilever fixed at one end with a point load at the other
problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0, 1.0), E, v, f)


# Optimization settings
V = 0.3    # Volume fraction
xmin = 1e-6 # Minimum density
rmin = 2.0  # Density filter radius

# Penalty for intermediate density values
penalty = TopOpt.PowerPenalty(3.0)

# Finite element solver setup
solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)

# Compliance objective
comp = TopOpt.Compliance(solver)
filter = DensityFilter(solver; rmin=rmin)
obj = x -> comp(filter(PseudoDensities(x)))

# Volume constraint
volfrac = TopOpt.Volume(solver)
constr = x -> volfrac(filter(PseudoDensities(x))) - V

# Initial design (volume fraction as initial guess for all elements)
x0 = fill(V, length(solver.vars))

# Define optimization model
model = Model(obj)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, constr)

# Optimization algorithm
alg = MMA87()
convcriteria = Nonconvex.KKTCriteria()
options = MMAOptions(; maxiter=3000, tol=Nonconvex.Tolerance(; x=1e-3, f=1e-3, kkt=0.001), convcriteria)

# Run the optimization
r = optimize(model, alg, x0; options)
npzwrite("my_arrays.npz", r.minimizer)


result_mesh = GeometryBasics.Mesh(problem, r.minimizer);

# Makie.mesh(result_mesh)


# Now visualize using Makie
fig = Makie.mesh(result_mesh)
Makie.display(fig)
show(fig)
# # Visualize the result
# fig = visualize(problem; topology=r.minimizer, default_exagg_scale=0.07, scale_range=10.0, vector_linewidth=3, vector_arrowsize=0.5)
# Makie.display(fig)

# Save the scene as an image
save("./fig.png", fig)