
# 1D regression.

import Random
import PyPlot

using LinearAlgebra
#import Interpolations
using Revise

include("../src/misc/declarations.jl")

include("../src/RKHS/RKHS.jl")
include("../src/RKHS/kernel.jl")

#include("../src/warp_map/Rieszanalysis.jl")
#include("../src/misc/utilities.jl")

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

PyPlot.close("all")

Random.seed!(25)

fig_num = 1

##### regression
θ = Spline34KernelType(0.2)

σ² = 1e-5
N = 15

# if 0 and 1 are included, we have posdef error, or rank = N - 2.
x_range = LinRange(1e-5, 1.0-1e-5, N)

X = collect( [x_range[n]] for n = 1:N )
coordwrapper = xx->convert2itpindex(xx, X[1], X[end], [length(X)])

f = xx->sinc(4*xx)*xx^3
y = f.(x_range)

# check posdef.
K = constructkernelmatrix(X, θ)
println("rank(K) = ", rank(K))

println("isposdef = ", isposdef(K))



# fit RKHS.
η = RKHSProblemType( zeros(Float64,length(X)),
                     X,
                     θ,
                     σ²)
fitRKHS!(η, y)



# query.
Nq = 100
xq_range = LinRange(0.0, 1.0, Nq)
xq = collect( [xq_range[n]] for n = 1:Nq )

f_xq = f.(xq_range) # generating function.

yq = Vector{Float64}(undef, Nq)
query!(yq,xq,η)

# Visualize regression result.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(X, y, ".", label = "observed")
PyPlot.plot(xq, yq, label = "query - RKHS")

PyPlot.plot(xq, f_xq, label = "true")

title_string = "1-D RKHS demo"
PyPlot.title(title_string)
PyPlot.legend()
##### end of regression
