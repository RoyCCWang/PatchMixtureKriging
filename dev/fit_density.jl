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

# import SignalTools
#
# import Utilities # https://gitlab.com/RoyCCWang/utilities
#
# import VisualizationTools # https://gitlab.com/RoyCCWang/visualizationtools


#import Calculus
#import ForwardDiff

import Utilities # https://gitlab.com/RoyCCWang/utilities
import ProbabilityUtilities # https://gitlab.com/RoyCCWang/probabilityutilities
import RiemannianOptim # https://gitlab.com/RoyCCWang/riemannianoptim

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



function visualizemeshgridpcolor(   x_ranges::Vector{LinRange{T,L}},
                                    Y::Matrix{T},
                                    marker_locations::Vector,
                                    marker_symbol::String,
                                    fig_num::Int,
                                    title_string::String;
                                    x1_title_string::String = "Dimension 1",
                                    x2_title_string::String = "Dimension 2",
                                    cmap = "Greens_r") where {T <: Real, L}
    #
    @assert length(x_ranges) == 2
    x_coords = collect( collect(x_ranges[d]) for d = 1:2 )

    PyPlot.figure(fig_num)
    fig_num += 1
    PyPlot.pcolormesh(x_coords[2], x_coords[1], Y, cmap = cmap)
    PyPlot.xlabel(x2_title_string)
    PyPlot.ylabel(x1_title_string)
    PyPlot.title(title_string)

    for i = 1:length(marker_locations)
        pt = reverse(marker_locations[i])
        PyPlot.annotate(marker_symbol, xy=pt, xycoords="data")
    end

    PyPlot.plt.colorbar()

    return fig_num
end


### I am here. write test script for fitnDdensityRiemannian. Then, move onto finite KR, helmt.

D = 2

N_components = 3
N_realizations = 100
N_array = [200; 200]

### set up coordinates.
τ = 15.0
limit_a = [-τ; -τ]
limit_b = [τ; τ]

x_ranges = collect( LinRange(limit_a[d], limit_b[d], N_array[d]) for d = 1:D )
X_nD = Utilities.ranges2collection(x_ranges, Val(D))
X = vec(X_nD)

### set up oracle density.
mixture_weights = rand(N_components)
mixture_weights = mixture_weights ./ sum(mixture_weights)

f, oracle_dist = ProbabilityUtilities.generaterandomGMM(mixture_weights, D)


# visualize full joint density.
f_X = f.(X)
f_X_nD = f.(X_nD)

fig_num = visualizemeshgridpcolor(x_ranges, f_X_nD, [], ".", fig_num, "oracle density, full joint, D = 2")


##### RKHS for density

zero_tol = 1e-13
max_iters = 500
σ² = 1e-3

optim_max_iters = 50000
show_trace_flag = false


########## second dimension.
println("Start working on d_select = 2")
d_select = 2

#######

𝑋 = collect( rand(oracle_dist) for n = 1:N_realizations)
𝑌 = f.(𝑋)

### fit density.
𝑌_nD = f.(X_nD)
σ² = 1e-5

# # warp map.
# amplification_factor = 1.0
# attenuation_factor_at_cut_off = 2
# N_bands = 5
# reciprocal_cut_off_percentages = ones(N_bands) ./collect(LinRange(1.0,0.2,N_bands))
# ω_set = collect( π/(reciprocal_cut_off_percentages[i]*sqrt(2*log(attenuation_factor_at_cut_off))) for i = 1:length(reciprocal_cut_off_percentages) )
# pass_band_factor = abs(ω_set[1]-ω_set[2])*0.2
#
# ϕ = getRieszwarpmapsamples(𝑌_nD, Val(:simple), Val(:uniform), ω_set, pass_band_factor)
#
# ϕ_map_func_2D, d_ϕ_map_func_2D, d2_ϕ_map_func_2D = Utilities.setupcubicitp(ϕ, x_ranges[1:d_select], amplification_factor)

θ_canonical = RationalQuadraticKernelType(0.7)
# θ_a_2D = AdaptiveKernelType(θ_canonical, ϕ_map_func_2D)
#
# println("Starting density fit: canonical kernel")
# @time η, α_SDP = fitnDdensity(vec(𝑌), vec(𝑋), σ², θ_canonical, zero_tol, max_iters)
# println("finished density fit")

# ## save data.
# import BSON
#
# y = vec(𝑌)
# X = vec(𝑋)
# println("construct matrix")
# @time K = constructkernelmatrix(X, θ_canonical)
# μ = σ²
# BSON.bson("density_fit_data1.bson", K = K,
#                                     y = y,
#                                     X = X,
#                                     μ = μ,
#                                     α_SDP = α_SDP)
# @assert 1==2

# ζ, α_optim, final_results,
#   initial_results_minimizer, ζ_costfunc = fitdensityoptim( vec(𝑌), vec(𝑋), σ², θ_canonical;
#                                         zero_tol = zero_tol,
#                                         max_iters = optim_max_iters,
#                                         show_trace = show_trace_flag )

#optim_max_iters = 5000
println("Starting density fit: canonical kernel, Riemannian")

# lower, faster, less accurate.
avg_Δf_tol = 1e-5 # 1e=10 gives better score than α_SDP.

@time ζ, α_optim = fitnDdensityRiemannian( vec(𝑌), vec(𝑋), σ²,
                                          θ_canonical, zero_tol, optim_max_iters;
                                          avg_Δf_tol = avg_Δf_tol )
println("finished density fit")
#
# discrepancy = norm(α_SDP - α_optim)
# println("SDP vs. Optim l-2 discrepancy: ", discrepancy)
#
K_costfunc = constructkernelmatrix(vec(𝑋), θ_canonical)
costfunc = aa->RiemannianOptim.RKHSfitdensitycostfunc(aa, K_costfunc, vec(𝑌), σ²)
#
# println("θ_canonical: costfunc(α_SDP) ", costfunc(α_SDP) )
println("θ_canonical: costfunc(α_optim) ", costfunc(α_optim) )
println()

# println("[α_SDP α_optim] = ")
# display([α_SDP α_optim])
# println()

# I am here. save another dataset, and tune the zero_tol of the retraction.
# finish up with swarm optim.

# TODO future works, or if insuccificent material for publication, fit conditional densities.

@assert 1==2


println("Starting density fit: adaptive kernel")
σ² = 1e-5
@time η_a, α_a_SDP = fitnDdensity(vec(𝑌), vec(𝑋), σ², θ_a_2D, zero_tol, max_iters)
println("finished density fit")


ζ_a, α_a_optim, final_results_a,
  initial_results_minimizer_a, ζ_costfunc_a = fitdensityoptim( vec(𝑌), vec(𝑋), σ², θ_a_2D;
                                            zero_tol = zero_tol,
                                            max_iters = optim_max_iters,
                                            show_trace = show_trace_flag )

discrepancy = norm(α_SDP - α_optim)
println("SDP vs. Optim l-2 discrepancy: ", discrepancy)
println("θ_canonical: costfunc(α_a_SDP) ", ζ_costfunc_a(α_a_SDP) )
println("θ_canonical: costfunc(α_a_optim) ", ζ_costfunc_a(α_a_optim) )
println()

# next, determined the kernel centers.

fq = xx->sum( η.c[n]*evalkernel(xx,η.X[n],η.θ) for n = 1:length(η.c) )


#
fq_a = xx->sum( η_a.c[n]*evalkernel(xx, η_a.X[n], η_a.θ) for n = 1:length(η_a.c) )


### visualize. Show adaptive kernel is useful.
# replace [] with 𝑋 to see kernel centers.
𝑌q = fq.(X_nD)
𝑌q_a = fq_a.(X_nD)

fig_num = visualizemeshgridpcolor(x_ranges[1:d_select], 𝑌q, [], "x",
                fig_num, "canonical query", "x1", "x2")

fig_num = visualizemeshgridpcolor(x_ranges[1:d_select], 𝑌q_a, [], "x",
                fig_num, "adaptive query", "x1", "x2")

###

gq = xx->sum( ζ.c[n]*evalkernel(xx,ζ.X[n],ζ.θ) for n = 1:length(ζ.c) )

#
gq_a = xx->sum( ζ_a.c[n]*evalkernel(xx, ζ_a.X[n], ζ_a.θ) for n = 1:length(ζ_a.c) )


### visualize. Show adaptive kernel is useful.
# replace [] with 𝑋 to see kernel centers.
gq_X = gq.(X_nD)
gq_a_X = gq_a.(X_nD)

fig_num = visualizemeshgridpcolor(x_ranges[1:d_select], gq_X,
                [], "x", fig_num, "canonical query: gq", "x1", "x2")

fig_num = visualizemeshgridpcolor(x_ranges[1:d_select], gq_a_X,
                [], "x", fig_num, "adaptive query: gq", "x1", "x2")
