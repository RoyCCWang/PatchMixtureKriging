
import Random
import PyPlot

using LinearAlgebra
import Interpolations

#import SignalTools

# include("../src/misc/declarations.jl")

# include("../src/RKHS/RKHS.jl")
# include("../src/RKHS/kernel.jl")
# #include("../src/RKHS/interpolators.jl")

# include("../src/warp_map/Rieszanalysis.jl")
# include("../src/misc/utilities.jl")

#include("../src/PatchMixtureKriging.jl")
import PatchMixtureKriging # Install command from Julia REPL: using Pkg; Pkg.add("https://github.com/RoyCCWang/PatchMixtureKriging")


PyPlot.close("all")

Random.seed!(25)

fig_num = 1

##### regression
#θ = Spline34KernelType(0.2)
θ = PatchMixtureKriging.BrownianBridge10(1.0)
#θ = BrownianBridge20(1.0)
#θ = BrownianBridge1ϵ(4.5)
#θ = BrownianBridge2ϵ(2.5)


σ² = 1e-5
N = 15

# if 0 and 1 are included, we have posdef error, or rank = N - 2.
x_range = LinRange(1e-5, 1.0-1e-5, N)

X = collect( [x_range[n]] for n = 1:N )
coordwrapper = xx->PatchMixtureKriging.convert2itpindex(xx, X[1], X[end], [length(X)])

f = xx->sinc(4*xx)*xx^3
y = f.(x_range)

# check posdef.
K = PatchMixtureKriging.constructkernelmatrix(X, θ)
println("rank(K) = ", rank(K))

println("isposdef = ", isposdef(K))



# fit RKHS.
η = PatchMixtureKriging.RKHSProblemType( zeros(Float64,length(X)),
                     X,
                     θ,
                     σ²)
                     PatchMixtureKriging.fitRKHS!(η, y)



# query.
Nq = 100
xq_range = LinRange(0.0, 1.0, Nq)
xq = collect( [xq_range[n]] for n = 1:Nq )

f_xq = f.(xq_range) # generating function.

yq = Vector{Float64}(undef, Nq)
PatchMixtureKriging.query!(yq,xq,η)

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
