
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

#PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

PyPlot.close("all")

Random.seed!(25)

fig_num = 1

D = 2
N = 500
X = collect( randn(D) for n = 1:N )
levels = 4 # 2^(levels-1) leaf nodes.

root = RKHSRegularization.setuppartition(X, levels)

# # print using AbstractTrees.
# AbstractTrees.printnode(io::IO, node::RKHSRegularization.BinaryNode) = print(io, node.data)
# AbstractTrees.print_tree(root)

### visualize tree.
# traverse all leaf nodes.
X_parts = Vector{Vector{Vector{Float64}}}(undef, 0)
RKHSRegularization.buildXpart!(X_parts, root)

function visualize2Dpartition(X_parts::Vector{Vector{Vector{T}}},
    fig_num::Int,
    title_string::String) where T <: Real
    
    PyPlot.figure(fig_num)
    fig_num += 1

    for i = 1:length(X_parts)

        # points.
        X = X_parts[i]

        x1 = collect( X[n][1] for n = 1:length(X))
        x2 = collect( X[n][2] for n = 1:length(X))

        PyPlot.scatter(x1, x2, label = "$(i)")
    end

    PyPlot.title(title_string)
    PyPlot.legend()

    return fig_num
end

fig_num = visualize2Dpartition(X_parts, fig_num, "X_parts")



y_set = Vector{Vector{Float64}}(undef, 0)
t_set = Vector{Vector{Float64}}(undef, 0)
min_t = -5.0
max_t = 5.0
max_N_t = 500
RKHSRegularization.getpartitionlines2D!(y_set, t_set, root, levels, min_t, max_t, max_N_t)