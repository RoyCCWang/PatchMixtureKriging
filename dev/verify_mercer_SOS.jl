# test mercer kernels and RKHS theory.


using LinearAlgebra
import Random

import GSL
import PyPlot
import Interpolations
import Distributions
using SpecialFunctions

import HCubature

import SignalTools


include("../src/misc/declarations.jl")

include("../src/RKHS/RKHS.jl")
include("../src/RKHS/kernel.jl")
include("../src/RKHS/interpolators.jl")
include("../src/RKHS/mercer.jl")

include("../src/warp_map/Rieszanalysis.jl")
include("../src/misc/utilities.jl")

PyPlot.close("all")

Random.seed!(25)

fig_num = 1

##### regression
θ = GaussianKernel1DType(1.1)

σ² = 1e-3
N = 14

x_range = LinRange(-π,2.34*π,N)
x_range = LinRange(-π-4, 2.34*π+4,N)

X = collect( [x_range[n]] for n = 1:N )

### specify oracle function.
c = randn(N)
f_hat = xx->sum( c[j]*evalkernel(xx, X[j], θ)^2 for j = 1:N )

# query locations.
Nq = 1000
xq_range = LinRange(-π-4, 2.34*π+2, Nq)
xq = collect( [xq_range[n]] for n = 1:Nq )



### set up mercer expansion.

# parameters.
M = 100 #  this is the truncation limit for the Mercer expansion.
α = 1.0

# pre-compute.
ϵ, β, δ_sq = setupmercer(θ, α)
λ_array = geteigenvalues(θ, M, α)

# unnormalized eigenfunction.
ϕ_tilde = (ii,xx)->sqrt(λ_array[ii])*evaleigenfunction(θ, α, β, δ_sq, ii, xx[1])

# normalized eigenfunction.
ϕ = (ii,xx)->evaleigenfunction(θ, α, β, δ_sq, ii, xx[1])

### verify Mercer expansion.
#x = randn()
#z = randn()
x = 11.0
z = 14.0

ϕ_x_array = collect( ϕ(m,x) for m = 1:M )
ϕ_z_array = collect( ϕ(m,z) for m = 1:M )

k_x_z = evalkernel([x],[z],θ)
mercer_sum = sum(λ_array[n]*ϕ_x_array[n]*ϕ_z_array[n] for n = 1:M)
println("k(x,z) = ", k_x_z, ", mercer sum = ", mercer_sum)
println("discrepancy = ", abs(k_x_z-mercer_sum) )
println()


### verify kernel to SOS.
Q = getQmat(c, ϕ_tilde, X, M)

f_hat2 = xx->sum( Q[i,j] *evaleigenfunction(θ, α, β, δ_sq, i, xx)
                *evaleigenfunction(θ, α, β, δ_sq, j, xx)
                *sqrt(λ_array[i])*sqrt(λ_array[j]) for i = 1:M for j = 1:M )

#
y_Q_SOS = f_hat2.(xq_range)
y_kernel_SOS = f_hat.(xq)
discrepancy = norm( y_Q_SOS - y_kernel_SOS )
println("discrepancy between kernel and Q SOS expressions = ", discrepancy)

# # Visualize regression result.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(xq_range, y_Q_SOS, ".", label = "y_Q_SOS")
PyPlot.plot(xq_range, y_kernel_SOS, label = "y_kernel_SOS")

title_string = "SOS RKHS - Gaussian Mercer series"
PyPlot.title(title_string)
PyPlot.legend()

# seems like for x > 9.4, things are so ill-conditioned that the kernel and
#   Q representations no longer agree. We could do log-space operations, but
#   this still mean the SDP problem, which involves a ill-conditioned kernel matrix,
#   is difficult to enforce to be posdef, at least numerically.


# continue investigating, just avoid using Q.


@assert 1==2

#### orthogonality check.
μ = xx->α/sqrt(π)*exp(-α^2*xx^2)
val, err = evalindefiniteintegral(μ)
println("integral of μ is ", val)

i = 70
j = 9
ϵ, β, δ_sq = setupmercer(θ, α)
#integrand_func = xx->μ(xx)*evaleigenfunction(θ, α, β, δ_sq, i, xx)*evaleigenfunction(θ, α, β, δ_sq, j, xx)
#val, err = evalindefiniteintegral(integrand_func)

#N_MC = 2000000
N_MC = 20000
σ = 1/(sqrt(2)*α)
measure_dist = Distributions.Normal(0.0, σ)
X = collect( rand(measure_dist) for n = 1:N_MC )
MC_func = xx->evaleigenfunction(θ, α, β, δ_sq, i, xx)*evaleigenfunction(θ, α, β, δ_sq, j, xx)

# check orthogonality of eigenfunctions wrt to the measure μ.
MC_integral = sum( MC_func(X[n])/N_MC for n = 1:N_MC)

### explore RKHS norm, and the nature of ℋ. Verdict: the norm is a series, not integration.
k_z = xx->evalkernel(xx, [z], θ)
g = xx->sum( η.c[j]*evalkernel(xx, η.X[j], θ) for j = 1:length(η.c) )

N_MC = 20000
MC_norm_g = sum( g( [X[n]] )^2/N_MC for n = 1:N_MC)

K = constructkernelmatrix(η.X, η.θ)
RKHS_norm_g = dot(η.c, K*η.c)
#RKHS_norm_g = sum( dot( η.c[j], g([η.X[j]]) ) for j = 1:length(η.c) )
println("RKHS norm(g) = ", RKHS_norm_g, ", MC_norm_g = ", MC_norm_g)




### explore SOS.
x = randn()
M = 100 #  this is the truncation limit for the Mercer expansion.
V = randn(M,M)
V = V./norm(V)
ϕ_x = evaleigenfunctions(θ, M, α, x)

f = xx->evaleigenfunctions(θ, M, α, xx)'*V'*V*evaleigenfunctions(θ, M, α, xx)
q = xx->evaleigenfunctions(θ, M, α, xx)

y = f.(x_range)
f_xq = f.(xq_range)

# # Visualize regression result.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(x_range, y, ".", label = "observed")
#PyPlot.plot(xq, yq, label = "query - RKHS")

PyPlot.plot(xq_range, f_xq, label = "true")

title_string = "SOS RKHS - Gaussian Mercer series"
PyPlot.title(title_string)
PyPlot.legend()
