
import Printf
import Random
using LinearAlgebra

import DifferentialEquations

import Distributions

import PyPlot

import StatsBase
import StatsFuns

import Printf

import Utilities # https://gitlab.com/RoyCCWang/utilities
import Calculus
import AdaptiveRKHS
import Statistics

include("../src/misc/declarations.jl")

include("../src/RKHS/RKHS.jl")
include("../src/RKHS/kernel.jl")
include("../src/RKHS/interpolators.jl")

include("../src/warp_map/Rieszanalysis.jl")
include("../src/misc/utilities.jl")

include("../src/RKHS/querying.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

D = 3

σ_oracle = 10.0
ρ_oracle = 28.0
β_oracle = 8/3

params_oracle = [σ_oracle; ρ_oracle; β_oracle]

# p is state vector.
function lorenz!(du::Vector{T}, u, p, t) where T
    σ = p[1]
    ρ = p[2]
    β = p[3]

    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]

    return nothing
end

tspan = (0.0, 100.0)

N_obs = 15 #30 #15
time_stamp_range = LinRange(0.1, tspan[end], N_obs)
time_stamp = collect(time_stamp_range)

u0 = zeros(Float64, D)
u0[1] = 1.0

prob = DifferentialEquations.ODEProblem(lorenz!, u0, tspan, params_oracle, dense = true)
sol = DifferentialEquations.solve(prob,
            DifferentialEquations.Tsit5(),
            reltol = 1e-8, abstol = 1e-8)
y_clean = sol.(time_stamp)


import Plots
Plots.plot(sol, vars = (1,2,3), show = true)

@assert 1==2

N_display = 1000
t_display = LinRange(tspan[1], tspan[2], N_display)


sol_x1_display = collect( sol.u[i][1] for i = 1:length(sol.u) )
sol_x2_display = collect( sol.u[i][2] for i = 1:length(sol.u) )
#y1_display = collect( y[i][d_select] for i = 1:length(y) )

t_dummy = 1.0



y_clean = y_clean #./ 100
σ = 5.0 #/100
y = collect( y_clean[i] + randn(D_state) .* σ for i = 1:length(y_clean) )


y1_clean = collect( y_clean[i][1] for i = 1:length(y_clean) )
y1_itp, dy1_itp, d2y1_itp = Utilities.setupcubicitp(y1_clean, [ time_stamp], 1.0)

dy1_ND = xx->Calculus.gradient(y1_itp, xx)[1]
dy1_itp_no_vec = xx->dy1_itp(xx)[1]
dy2_itp_no_vec = xx->d2y1_itp(xx)[1]



### try RKHS solution since.
θ_oracle = AdaptiveRKHS.GaussianKernel1DType(1.0/100)

σ²_oracle = 1e-7

X_oracle = collect( [sol.t[i]] for i = 1:length(sol.t) )

u_array = Vector{Vector{Float64}}(undef,2)
u_array[1] = collect( sol.u[i][1] for i = 1:length(sol.u) )
u_array[2] = collect( sol.u[i][2] for i = 1:length(sol.u) )

u_GP = fitDEsolutionGP(θ_oracle, σ²_oracle, X_oracle, u_array)
u1_GP = u_GP[1]
u2_GP = u_GP[2]

du_GP = tt->evalpreyderivativeswrttime(u1_GP, u2_GP, p_oracle, tt)

du1_GP = tt->du_GP(tt)[1]
du1_GP_ND = tt->Calculus.gradient(u1_GP, tt)[1]
#d2u1_GP_ND = tt->Calculus.hessian(u1_GP, tt)[1]
d2u1_GP = tt->Calculus.gradient(du1_GP, tt)[1]

du2_GP = tt->du_GP(tt)[1]
du2_GP_ND = tt->Calculus.gradient(u2_GP, tt)[1]

Nq = 500
# xq_range = LinRange(sol.t[1], sol.t[end], Nq)
xq_range = LinRange(time_stamp[1], time_stamp[end], Nq)
xq = collect( [ xq_range[i] ] for i = 1:length(xq_range) )



title_string = Printf.@sprintf("Predator-prey populations")

PyPlot.figure(fig_num)
fig_num += 1
PyPlot.plot(sol.t, sol_x1_display, label = "species 1")
#PyPlot.plot(sol.t, sol_x2_display, label = "species 2")
PyPlot.plot(xq, y1_itp.(xq), label = "y1 itp")
PyPlot.plot(xq, u1_GP.(xq), "x", label = "u1 GP")
PyPlot.title(title_string)
PyPlot.legend()

@assert 1==2

title_string = "derivative"
PyPlot.figure(fig_num)
fig_num += 1
#PyPlot.plot(xq, dy1_ND.(xq), label = "numerical itp")
#PyPlot.plot(xq, dy1_itp_no_vec.(xq), "x", label = "analytical itp")
PyPlot.plot(xq, du1_GP_ND.(xq), "x", label = "numerical GP")
PyPlot.plot(xq, du1_GP.(xq), label = "analytical GP")
PyPlot.title(title_string)
PyPlot.legend()

#@assert 1==2

println("norm(dy1_ND.(xq) - dy1_itp_no_vec.(xq)) = ", norm(dy1_ND.(xq) - dy1_itp_no_vec.(xq)))
println()

# fit RKHS.


θ = AdaptiveRKHS.GaussianKernel1DType(1.0/100)

σ²_RKHS_initial = sqrt(σ) #σ^2 #1e-3

X = collect( [time_stamp_vec[i]] for i = 1:length(time_stamp_vec) )
y1 = collect( y[i][1] for i = 1:length(y) )
η = AdaptiveRKHS.RKHSProblemType( zeros(Float64,length(X)),
                     X,
                     θ,
                     σ²_RKHS_initial)
offset_c = Statistics.mean(y1)
y1_c = y1 .- offset_c
AdaptiveRKHS.fitRKHS!(η, y1_c)

fq = xx->sum( η.c[i]*AdaptiveRKHS.evalkernel(xx, η.X[i], η.θ) for i = 1:length(η.X) )+offset_c


# adaptive kernel.
θ_canonical = AdaptiveRKHS.GaussianKernel1DType(1.0/1000)
#warp map.
# d2f_normed = xx->norm(Calculus.hessian(f_joint,xx))
# d2f_x1 = xx->Calculus.hessian(f_joint,xx)[1,1]
# d2f_x2 = xx->Calculus.hessian(f_joint,xx)[2,2]
# d2f_sum = xx->sum(Calculus.hessian(f_joint,xx))

#d2y1 = xx->Calculus.hessian(fq,xx)[1]
#d2y1 = xx->Calculus.hessian(y1_itp,xx)[1]
d2y1 = xx->Calculus.hessian(u1_GP,xx)[1]
#ϕ = d2y1 #getwarpmap(ϕ)
ϕ = d2u1_GP

title_string = "second derivative"
PyPlot.figure(fig_num)
fig_num += 1
#PyPlot.plot(xq, d2u1_GP_ND.(xq), label = "d2u1_GP_ND")
PyPlot.plot(xq, d2u1_GP.(xq), "^", label = "d2u1_GP")
PyPlot.title(title_string)
PyPlot.legend()

# adaptive RKHS.
M = 30
#σ²_RKHS_array = sqrt(σ) .* ones(Float64, M) # #1.0 #1e-1 #1e-3
σ²_RKHS_array = collect( LinRange(sqrt(σ), σ^2, M) )
amp_factor_array = 100.0 .* ones(Float64, M)

fq_array, c_array, θ_array = iterateGP(fq, ϕ, θ_canonical, σ²_RKHS_array, amp_factor_array, y1, X)

#### visualize.

fq_xq = fq.(xq)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(time_stamp, y1, "o", label = "observed")
PyPlot.plot(xq, fq_xq, label = "canonical RKHS")
PyPlot.plot(xq, fq_array[1].(xq), label = "adaptive RKHS")
PyPlot.plot(xq, fq_array[2].(xq), label = "adaptive RKHS 2")
PyPlot.plot(xq, fq_array[3].(xq), label = "adaptive RKHS 3")
PyPlot.plot(xq, fq_array[end].(xq), label = "adaptive RKHS end")
PyPlot.plot(sol.t, sol_x1_display, "--", label = "species 1")
#PyPlot.plot(sol.t, sol_x2_display, label = "species 2")

title_string = "1-D RKHS demo"
PyPlot.title(title_string)
PyPlot.legend()
