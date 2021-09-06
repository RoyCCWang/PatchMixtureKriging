# demo sum-of-squares RKHS.

using LinearAlgebra
import Random

import PyPlot
import Interpolations
import Distributions

import SignalTools


include("../src/misc/declarations.jl")

include("../src/RKHS/RKHS.jl")
include("../src/RKHS/kernel.jl")
include("../src/RKHS/interpolators.jl")

include("../src/warp_map/Rieszanalysis.jl")
include("../src/misc/utilities.jl")

PyPlot.close("all")

Random.seed!(25)

fig_num = 1

##### regression

# use spline (better conditioned) for large N.
#θ = Spline34KernelType(0.2)
#θ = Spline34KernelType(0.6)
#θ = Spline34KernelType(0.8)
θ = Spline34KernelType(1.1)

# use this kernel when doing investigation on the eigenfunctions that are associated
#   with the kernel.
#θ = GaussianKernel1DType(1.1)
#θ = GaussianKernel1DType(0.8)

μ = 1e-6
#μ = 1.0
#μ = 0.0

#N = 14
#N = 50
N = 500

x_range = LinRange(-π,2.34*π,N)
x_range = LinRange(-π-4, 2.34*π+4,N)

X = collect( [x_range[n]] for n = 1:N )
coordwrapper = xx->convert2itpindex(xx, X[1], X[end], [length(X)])

### specify oracle function.

K = 3
mixture_weights = collect( sqrt(n) for n = 1:K)
mixture_weights = mixture_weights./sum(mixture_weights)

normal_dists = collect( Distributions.Normal(randn(), 2*rand()) for n = 1:K )

# mixture distribution.
mix_normal_dist = Distributions.MixtureModel(normal_dists, mixture_weights)

f = xx->sinc(xx)*xx^3
#f = xx->10*Distributions.pdf(mix_normal_dist,xx)
y = f.(x_range)


### SDP
max_iterations = 50000
#max_iterations = 5000

# Make the Convex.jl module available
using Convex
using SCS
using SDPA

# Generate random problem data


I_mat = Matrix{Float64}(LinearAlgebra.I, N, N)
K = constructkernelmatrix(X, θ)
Kp = K.^2
#G = K'*K
G = K

M_mat = Kp'*Kp + μ*Kp
L_mat = cholesky(M_mat).U
N_mat = -2*Kp'*y
zeros_N = zeros(Float64,N,N)
zeros_1N = zeros(Float64,1,N)

# Create a (column vector) variable of size n x 1.
t = Variable()
α = Variable(N)

diag_α = diagm(0=>α)


# constraint.
# constraint = ([ I_mat L*α zeros_n;
#                 α'*L' t+2*dot(Kp*α,y) zeros_1n;
#                 zeros_n zeros_1n' G*diag_α+diag_α*G] ⪰ 0)
#
# problem = minimize(t+1-1, constraint)

# f is the square of a function.
#constraints = collect( (G*α)[i] >= 0 for i = 1:N )
constraints = collect( α[i] >= 0 for i = 1:N )
obj_expr = ( sumsquares(Kp*α-y) + μ*quadform(α, Kp) )
problem = minimize(obj_expr, constraints)

# f is the sum of the squares of functions, but each q_i isn't an inf sum.
# constraints = collect( (dot(G[i,:],α) >= 0) for i = 1:N )
# obj_expr = ( dot(Kp*α-y, Kp*α-y) + μ*dot(α, Kp*α) )
# problem = minimize(obj_expr, constraints[1])

# The problem is to minimize ||Ax - b||^2 subject to x >= 0
# This can be done by: minimize(objective, constraints)


# Solve the problem by calling solve!
#@time solve!(problem, SCSSolver(max_iters = max_iterations))
@time solve!(problem, SCS.Optimizer(max_iters = max_iterations))
#@time solve!(problem, SDPASolver(MaxIteration = max_iterations))


# Check the status of the problem
problem.status # :Optimal, :Infeasible, :Unbounded etc.
println("status: ", problem.status)

# Get the optimum value
problem.optval

α_star = vec(α.value)

θ_sos = SOSKernelType(θ)
η = RKHSProblemType(α_star, X, θ_sos, μ)

### experiment: modify α_star.
#clamp!(α_star, -0.1, Inf)
#α_star[7] = -0.1
### end experimentation.

# query.
Nq = 1000
xq_range = LinRange(-π-4, 2.34*π+4, Nq)
xq = collect( [xq_range[n]] for n = 1:Nq )

f_xq = f.(xq_range) # generating function.

yq = Vector{Float64}(undef, Nq)
query!(yq,xq,η)

# Visualize regression result.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(X, y, ".", label = "observed")
PyPlot.plot(xq, yq, "x", label = "query - RKHS")

PyPlot.plot(xq, f_xq, label = "true")

title_string = "SOS-RKHS demo"
PyPlot.title(title_string)
PyPlot.legend()

println("All (numerically) strictly positive values in α_star: ", all(α_star.>0))
println("minimum(α_star) = ", minimum(α_star))
println("All (numerically) strictly positive values in yq: ", all(yq.>0))
println("minimum(yq) = ", minimum(yq))

U = G*diagm(0=>α_star)
s,Q_U = eigen(U)

# verify Sylvester's law: congruent sym real matrices have same eigenvalue signs.
min_val, l_select = findmin(yq)
k_evals_l = collect( α_star[i]*evalkernel(xq[l_select], X[i], θ) for i = 1:N )
yq_l = sum(k_evals_l)
println("yq[l_select]-yq_l = ", yq[l_select]-yq_l)
