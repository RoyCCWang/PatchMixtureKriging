import Random
import PyPlot
import Statistics

using LinearAlgebra
#import Interpolations
#using Revise

include("../src/RKHSRegularization.jl")
import .RKHSRegularization # https://github.com/RoyCCWang/RKHSRegularization

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])
import Random
import PyPlot
import Statistics

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
N = 200
PyPlot.close("all")

Random.seed!(25)

fig_num = 1

D = 2
N = 200
X = randn(D,N)

U, s, V = svd(X')

u = V[:,1]

X_set = collect( X[:,n] for n = 1:size(X,2) )
flags, functional_evals, c = RKHSRegularization.splitpoints(u, X_set)
X1 = X[:, flags]
X2 = X[:, .!flags]


# dot(u,x)+c = 0 to y = m*x + b.
function get2Dline(u::Vector{T}, c::T) where T

    m = -u[1]/u[2]
    b = c/u[2]

    return m, b
end

m, b = get2Dline(u, c)
t = LinRange(-5, 5, 300)
y = m .* t .+ b

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.scatter(X1[1,:], X1[2,:], label = "X1")
PyPlot.scatter(X2[1,:], X2[2,:], label = "X2")
PyPlot.plot(t, y)

title_string = "split points"
PyPlot.title(title_string)
PyPlot.legend()