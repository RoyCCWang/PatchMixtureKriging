
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

PyPlot.close("all")

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])


Random.seed!(25)

fig_num = 1

θ = RKHSRegularization.Spline34KernelType(1.0)
D = 1
σ² = 1e-5
N = 32

# if 0 and 1 are included, we have posdef error, or rank = N - 2.
x_range = LinRange(-1, 2, N)

X = collect( [x_range[n]] for n = 1:N )

f = xx->sinc(4*xx)*xx^3
y = f.(x_range)


K = RKHSRegularization.constructkernelmatrix(X, θ)

N_parts = 5
X_parts = collect( collect( randn(D) for n = 1:N ) for n = 1:N_parts )

B = RKHSRegularization.PatchGPType{typeof(θ), Float64}(X_parts, θ)

A = RKHSRegularization.setuppatchGP!(B)

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