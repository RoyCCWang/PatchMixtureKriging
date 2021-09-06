# final scheme. adaptive kernel.

import Printf
import PyPlot
import Random
import Optim

using LinearAlgebra

using FFTW

import Statistics

import Distributions
#import HCubature
import Interpolations
import SpecialFunctions

import SignalTools

import Utilities

import VisualizationTools
import ProbabilityUtilities

import Convex
import SCS

import Calculus
import ForwardDiff

import RiemannianOptim

include("../src/misc/declarations.jl")

include("../src/RKHS/RKHS.jl")
include("../src/RKHS/kernel.jl")
include("../src/RKHS/interpolators.jl")

include("../src/warp_map/Rieszanalysis.jl")
include("../src/misc/utilities.jl")

include("../src/optimization/sdp.jl")


include("../src/misc/front_end.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)


D = 2
# https://github.com/rdeits/RegionTrees.jl/blob/master/examples/demo/demo.ipynb

import Images, TestImages

rgb_image = TestImages.testimage("lighthouse")
gray_image = Images.Gray.(rgb_image)
#Images.mosaicview(rgb_image, gray_image; nrow = 1)

imA = convert(Matrix{Float64}, gray_image)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.imshow(imA, cmap = "gray" )

PyPlot.title("oracle image")
PyPlot.legend()


down_factor = 5
imB = imA[1:down_factor:size(imA,1), 1:down_factor:size(imA,2)]

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.imshow(imB, cmap = "gray" )

PyPlot.title("input image")
PyPlot.legend()


function assemblepatchs(patch_inds::Matrix{Tuple{UnitRange{Int},UnitRange{Int}}},
    X::Matrix{Vector{T}},
    y::Matrix{T}) where T

    X_set = Matrix{Matrix{Vector{T}}}(undef, size(patch_inds))
    y_set = Matrix{Matrix{T}}(undef, size(patch_inds))

    for r = 1:size(patch_inds,1)
        for c = 1:size(patch_inds,2)

            inds = patch_inds[r,c]
            # println("inds = ", inds)
            # println("X[inds[1], inds[2]] = ", X[inds[1], inds[2]])

            X_set[r,c] = X[inds[1], inds[2]]
            y_set[r,c] = y[inds[1], inds[2]]
        end
    end

    return X_set, y_set
end

function getpatches(Y::Matrix{T}, patch_sz::Tuple{Int,Int}; half_overlap::Int = 2) where T

    N_rows, N_cols = size(Y)
    #V = collect( Iterators.product(1:1:N_rows, 1:1:N_cols) )

    M_rows = ceil(Int, size(Y,1)/patch_sz[1])
    M_cols = ceil(Int, size(Y,2)/patch_sz[2])

    patch_inds = Matrix{Tuple{UnitRange{Int},UnitRange{Int}}}(undef, M_rows, M_cols)
    for r = 1:M_rows
        for c = 1:M_cols


            r_st = max(1, 1+(r-1)*patch_sz[1] - half_overlap)
            r_fin = min(N_rows, r*patch_sz[1] + half_overlap)

            c_st = max(1, 1+(c-1)*patch_sz[1] - half_overlap)
            c_fin = min(N_cols, c*patch_sz[1] + half_overlap)

            if r == M_rows
                r_fin = N_rows
            end

            if c == M_cols
                c_fin = N_cols
            end

            #patch_inds[r,c] = V[r_st:r_fin, c_st:c_fin]
            patch_inds[r,c] = (r_st:r_fin, c_st:c_fin)
        end
    end

    return patch_inds
end

function fitRKHSpathces(imB::Matrix{T}, θ, σ², patch_inds) where T

    N_rows, N_cols = size(imB)

    # tall.
    x1 = 1#one(T)
    x2 = N_cols/N_rows

    # wide.
    if N_cols > N_rows
        x1 = N_rows/N_cols
        x2 = 1#one(T)
    end

    X0 = collect( Iterators.product(LinRange(0, x1, N_rows), LinRange(0, x2, N_cols)) )
    X = reshape(collect( collect(X0[n]) for n = 1:length(X0) ), size(X0))

    X_set, y_set = assemblepatchs(patch_inds, X, imB)

    η_set = Matrix{RKHSProblemType}(undef, size(patch_inds))

    for r = 1:size(patch_inds,1)
        for c = 1:size(patch_inds,2)
            η_set[r,c] = RKHSProblemType( zeros(T, length(X_set[r,c])),
                vec(X_set[r,c]), θ, σ²)

            fitRKHS!(η_set[r,c], vec(y_set[r,c]))
        end
    end

    return η_set, X_set, y_set
end

patch_sz = (20,20)
patch_inds = getpatches(imB, patch_sz)

σ² = 1e-5
θ = Spline34KernelType(0.2)

println("Training RKHS.")
@time η_set, X_set, y_set = fitRKHSpathces(imB, θ, σ², patch_inds)


function querypatches(N_rows::Int, N_cols::Int, upfactor::Int,
    η_set, patch_inds) where T


    for r = 1:size(η_set,1)
        for c = 1:size(η_set,2)

            x1_st = X_set[r,c][1,1]
            x1_fin = X_set[r,c][end,end]
            N_x1 = size(X_set[r,c],1)

            x2_st = X_set[r,c][1,1]
            x2_fin = X_set[r,c][end,end]
            N_x2 = size(X_set[r,c],2)

            Xq0 = collect( Iterators.product(LinRange(x1_st, x1_fin, N_x1*upfactor),
            LinRange(x2_st, x2_fin, N_x2*upfactor)) )

            Xq = collect( collect(Xq0[n]) for n = 1:length(Xq0) )

            yq = Vector{T}(undef, length(Xq))
            query!(yq, Xq, η_set[r,c])

            out[r,c] = reshape(yq, N_x1*upfactor, N_x2*upfactor)
        end
    end

    return η_set, X_set, y_set
end

# query.
upfactor = 2
Xq0 = collect( Iterators.product(LinRange(0, x1, upfactor*N_rows), LinRange(0, x2, upfactor*N_cols)) )
Xq = collect( collect(Xq0[n]) for n = 1:length(Xq0) )

yq = Vector{Float64}(undef, length(Xq))
println("Querying RKHS.")
@time query!(yq, Xq, η)

imC = reshape(yq, (upfactor*N_rows,upfactor*N_cols))


PyPlot.figure(fig_num)
fig_num += 1

PyPlot.imshow(imC, cmap = "gray" )

PyPlot.title("output image")
PyPlot.legend()
