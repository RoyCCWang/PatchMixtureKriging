
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


include("../examples/helpers/visualization.jl")


PyPlot.close("all")

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])


Random.seed!(25)

fig_num = 1

θ = RKHSRegularization.Spline34KernelType(1.0)
D = 2
σ² = 1e-5
N = 850

# if 0 and 1 are included, we have posdef error, or rank = N - 2.
#x_range = LinRange(-1, 2, N)



# define the oracle.
f = xx->sinc((norm(xx)/3.2)^2)*(norm(xx)/4)^3

#### visualize the oracle.
N_array = [100; 200] # number of samples x1, x2.
limit_a = [-5.0; -10.0] # min x1, x2.
limit_b = [5.0; 10.0] # max x1, x2.
x_ranges = collect( LinRange(limit_a[d], limit_b[d], N_array[d]) for d = 1:D )

#X_nD = Utilities.ranges2collection(x_ranges, Val(D)) # x1 is vertical.

# force x1 to be horizontal instead of vertical.
X_nD = Utilities.ranges2collection(reverse(x_ranges), Val(D)) # x1 is horizontal.

f_X_nD = f.(X_nD)

#fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges,
fig_num = visualizemeshgridpcolorx1horizontal(x_ranges,
f_X_nD, [], "x", fig_num, "Oracle")
# top left is origin, Downwards is positive x2, To the right is positive x1.

#### sample from oracle.

#X = collect( randn(D) for n = 1:N )

X = collect( [Utilities.convertcompactdomain(rand(), 0.0, 1.0, limit_a[1], limit_b[1]);
Utilities.convertcompactdomain(rand(), 0.0, 1.0, limit_a[2], limit_b[2])] for n = 1:N )
y = f.(X)




levels = 3 # 2^(levels-1) leaf nodes. Must be larger than 1.

root, X_parts, X_parts_inds = RKHSRegularization.setuppartition(X, levels)

# following should be the same.
y_parts = collect( y[X_parts_inds[n]] for n = 1:length(X_parts_inds))




# @assert 1==4

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

fig_num, ax = visualize2Dpartition(X_parts, y_set, t_set, fig_num, "levels = $(levels)")
PyPlot.axis("scaled")

ax[:set_xlim]([limit_a[1],limit_b[1]]) # x1 is horizontal (x).
ax[:set_ylim]([limit_a[2],limit_b[2]]) # x2 is vertical (y).

#@assert 5==4

#K = RKHSRegularization.constructkernelmatrix(X, θ)

#N_parts = 5
#X_parts = collect( collect( randn(D) for n = 1:N ) for n = 1:N_parts )

#B = RKHSRegularization.mixtureGPType{typeof(θ), Float64}(X_parts, θ)
hps = RKHSRegularization.fetchhyperplanes(root)
η = RKHSRegularization.MixtureGPType(X_parts, hps)

# fit RKHS.
println("Begin query:")
@time RKHSRegularization.fitmixtureGP!(η, y_parts, θ, σ²)
println()

# query.
Xq = vec(X_nD)

# resize!(Xq, 1)
# Xq[1] = [-0.02; 6.7]
# Xq[1] = [-10.0, -5.0]
# Xq[1] = [-2.58, 9.13]

Yq = Vector{Float64}(undef, 0)
Vq = Vector{Float64}(undef, 0)

# for RBF profile.
weight_θ = RKHSRegularization.Spline34KernelType(1.0)
#RKHSRegularization.evalkernel(0.9, weight_θ)

radius = 0.3
δ = 1e-5
RKHSRegularization.querymixtureGP!(Yq, Vq, Xq, η, root, levels, radius, δ, θ, σ², weight_θ)

q_X_nD = reshape(Yq, size(f_X_nD))

fig_num = visualizemeshgridpcolorx1horizontal(x_ranges,
q_X_nD, [], "x", fig_num, "Yq")

# I am here. figure out the coordinate system once and for all. then, figure out why
# there are gaps along the boundary.
# then, compare with full GP.