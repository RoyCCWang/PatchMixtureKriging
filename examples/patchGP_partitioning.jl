
# each leaf node is a partition.

import Random
import PyPlot
import Statistics

import AbstractTrees

using LinearAlgebra
#import Interpolations
#using Revise

import Colors

#include("../src/RKHSRegularization.jl")
import RKHSRegularization # https://github.com/RoyCCWang/RKHSRegularization

#PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

include("./helpers/visualization.jl")

PyPlot.close("all")

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])


Random.seed!(25)

fig_num = 1


### plotting.


D = 2
N = 500
X = collect( randn(D) for n = 1:N )
levels = 2 # 2^(levels-1) leaf nodes. Must be larger than 1.

root, X_parts, _ = RKHSRegularization.setuppartition(X, levels)

# # print using AbstractTrees.
# AbstractTrees.printnode(io::IO, node::RKHSRegularization.BinaryNode) = print(io, node.data)
# AbstractTrees.print_tree(root)

### visualize tree.
# traverse all leaf nodes.
# X_parts = Vector{Vector{Vector{Float64}}}(undef, 0)
# RKHSRegularization.buildXpart!(X_parts, root)


centroid = Statistics.mean( Statistics.mean(X_parts[i]) for i = 1:length(X_parts) )
max_dist = maximum( maximum( norm(X_parts[i][j]-centroid) for j = 1:length(X_parts[i])) for i = 1:length(X_parts) ) * 1.1

y_set = Vector{Vector{Float64}}(undef, 0)
t_set = Vector{Vector{Float64}}(undef, 0)
min_t = -5.0
max_t = 5.0
max_N_t = 5000
RKHSRegularization.getpartitionlines!(y_set, t_set, root, levels, min_t, max_t, max_N_t, centroid, max_dist)

fig_num, _ = visualize2Dpartition(X_parts, y_set, t_set, fig_num, "levels = $(levels)")




levels = 3 # 2^(levels-1) leaf nodes. Must be larger than 1.

root, X_parts, _ = RKHSRegularization.setuppartition(X, levels)

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

fig_num, _ = visualize2Dpartition(X_parts, y_set, t_set, fig_num, "levels = $(levels)")



levels = 4 # 2^(levels-1) leaf nodes. Must be larger than 1.

root, X_parts, _ = RKHSRegularization.setuppartition(X, levels)

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

fig_num, ax4 = visualize2Dpartition(X_parts, y_set, t_set, fig_num, "levels = $(levels)")


levels = 5 # 2^(levels-1) leaf nodes. Must be larger than 1.

root, X_parts, _ = RKHSRegularization.setuppartition(X, levels)

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



#### test partition search given points. Write as unit stress test later.

x = [0.42; 2.05] # leaf node 11
x = [3.99; 3.35] # leaf node 16
x = [0.1; 0.16] # leaf node 9
x_region_ind = RKHSRegularization.findpartition(x, root, levels)


#


#@assert 1==2

####### point, find all hyperplanes intersection with sphere.

levels = 4 # 2^(levels-1) leaf nodes. Must be larger than 1.

root, X_parts, _ = RKHSRegularization.setuppartition(X, levels)

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


### work on a training set that is different than the partition.
ε = 0.3
X_set, X_set_inds, region_list_set,
problematic_inds = RKHSRegularization.organizetrainingsets(root, levels, X, ε)

fig_num = visualizesingleregions(X_set, y_set, t_set, fig_num)

N2 = collect( length(X_set[n]) for n = 1:length(X_set) )
N1 = collect( length(X_parts[n]) for n = 1:length(X_parts) )
println("N: X_parts: ", N1)
println("N: X_set:   ", N2)
println()



#p = randn(D) # test point.
p = x

# find the region for p.
p_region_ind = RKHSRegularization.findpartition(p, root, levels)
println("p_region_ind = ", p_region_ind)
println()

# get all hyperplanes.
hps = RKHSRegularization.fetchhyperplanes(root)
@assert length(hps) == length(X_parts) - 1 # sanity check.

radius = ε #0.3
δ = 1e-5
region_inds, ts, zs, hps_keep_flags = RKHSRegularization.findneighbourpartitions(p, radius, root, levels, hps, p_region_ind; δ = δ)

# debug.
hps_kept = hps[hps_keep_flags]
hp = hps_kept[1]
u = hp.v
c = hp.c

z_kept = zs[hps_keep_flags]
t_kept = ts[hps_keep_flags]
#z = z_kept[1]

dists = collect( norm(z_kept[i]-p) for i = 1:length(z_kept) )
@assert norm( dists - abs.(t_kept) ) < 1e-10 # sanity check.

# visualize on fresh plot.
fig_num, ax = visualize2Dpartition(X_parts, y_set, t_set, fig_num, "levels = $(levels)")

PyPlot.scatter([p[1];], [p[2];], marker = "x", s = 600.0, label = "p")

for i = 1:length(z_kept)
    PyPlot.scatter([z_kept[i][1];], [z_kept[i][2];], marker = "d", s = 600.0, label = "z = $(i)")
end

PyPlot.axis([p[1]-radius; p[1]+radius; p[2]-radius; p[2]+radius])
PyPlot.axis("scaled")

PyPlot.legend()
