
import Random
import PyPlot
import Statistics

import AbstractTrees

using LinearAlgebra
#import Interpolations
#using Revise

import Colors

include("../src/RKHSRegularization.jl")
import .RKHSRegularization # https://github.com/RoyCCWang/RKHSRegularization

#PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

import Utilities
import VisualizationTools

PyPlot.close("all")

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])


Random.seed!(25)

fig_num = 1

θ = RKHSRegularization.Spline34KernelType(1.0)
D = 2
σ² = 1e-5
N = 32

# if 0 and 1 are included, we have posdef error, or rank = N - 2.
#x_range = LinRange(-1, 2, N)



# define the oracle.
f = xx->sinc((norm(xx)/3.2)^2)*(norm(xx)/4)^3

#### visualize the oracle.
N_array = [100; 100]
limit_a = [-5.0; -5.0]
limit_b = [5.0; 5.0]
x_ranges = collect( LinRange(limit_a[d], limit_b[d], N_array[d]) for d = 1:D )
X_nD = Utilities.ranges2collection(x_ranges, Val(D))

f_X_nD = f.(X_nD)

fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges,
f_X_nD, [], "x", fig_num, "Oracle")

#### sample from oracle.

X = collect( randn(D) for n = 1:N )
y = f.(X)




levels = 3 # 2^(levels-1) leaf nodes. Must be larger than 1.

root, X_parts, X_parts_inds = RKHSRegularization.setuppartition(X, levels)

# following should be the same.
U1 = X_parts[3]
U2 = X[X_parts_inds[3]]


@assert 1==4

# # print using AbstractTrees.
# AbstractTrees.printnode(io::IO, node::RKHSRegularization.BinaryNode) = print(io, node.data)
# AbstractTrees.print_tree(root)

### visualize tree.
centroid = Statistics.mean( Statistics.mean(X_parts[i]) for i = 1:length(X_parts) )
max_dist = maximum( maximum( norm(X_parts[i][j]-centroid) for j = 1:length(X_parts[i])) for i = 1:length(X_parts) ) * 1.1

y_set = Vector{Vector{Float64}}(undef, 0)
t_set = Vector{Vector{Float64}}(undef, 0)
min_t = -5.0
max_t = 5.0
max_N_t = 5000
RKHSRegularization.getpartitionlines!(y_set, t_set, root, levels, min_t, max_t, max_N_t, centroid, max_dist)

# fig_num = visualize2Dpartition(X_parts, y_set, t_set, fig_num, "levels = $(levels)")


@assert 5==4

K = RKHSRegularization.constructkernelmatrix(X, θ)

N_parts = 5
X_parts = collect( collect( randn(D) for n = 1:N ) for n = 1:N_parts )

#B = RKHSRegularization.mixtureGPType{typeof(θ), Float64}(X_parts, θ)

y_parts = collect( TODO for n = 1:N_parts )
A = RKHSRegularization.setupmixtureGP(X_parts, y_parts, θ, σ²)

@assert 1==2

# check posdef.

println("rank(K) = ", rank(K))

println("isposdef = ", isposdef(K))



# fit RKHS.
η = RKHSRegularization.RKHSProblemType( zeros(Float64,length(X)),
                     X,
                     θ,
                     σ²)
RKHSRegularization.fitRKHS!(η, y)



# query.
Nq = 1000
xq_range = LinRange(x_range[1], x_range[end], Nq)
xq = collect( [xq_range[n]] for n = 1:Nq )

f_xq = f.(xq_range) # generating function.

yq = Vector{Float64}(undef, Nq)
RKHSRegularization.query!(yq,xq,η)

# Visualize regression result.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(X, y, ".", label = "observed")
PyPlot.plot(xq, yq, "--", label = "query - RKHS")

PyPlot.plot(xq, f_xq, label = "true")

title_string = "1-D RKHS demo"
PyPlot.title(title_string)
PyPlot.legend()