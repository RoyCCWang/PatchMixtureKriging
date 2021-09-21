
# each leaf node is a partition.

import Random
import PyPlot
import Statistics

import AbstractTrees

using LinearAlgebra
#import Interpolations
#using Revise

include("../src/RKHSRegularization.jl")
import .RKHSRegularization # https://github.com/RoyCCWang/RKHSRegularization

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

PyPlot.close("all")

Random.seed!(25)

fig_num = 1

D = 2
N = 500
X = collect( randn(D) for n = 1:N )
num_levels = 4 # 2^(num_levels-1) leaf nodes.

root = RKHSRegularization.setuppartition(X, num_levels)

# # print using AbstractTrees.
# AbstractTrees.printnode(io::IO, node::RKHSRegularization.BinaryNode) = print(io, node.data)
# AbstractTrees.print_tree(root)

### visualize tree.
# traverse all leaf nodes.
X_parts = Vector{Vector{Vector{Float64}}}(undef, 0)
RKHSRegularization.buildXpart!(X_parts, root)

# PyPlot.figure(fig_num)
# fig_num += 1

# PyPlot.scatter(X1[1,:], X1[2,:], label = "X1")
# PyPlot.scatter(X2[1,:], X2[2,:], label = "X2")
# PyPlot.plot(t, y)

# title_string = "split points"
# PyPlot.title(title_string)
# PyPlot.legend()