

function array2matrix(X::Vector{Vector{T}})::Matrix{T} where T

    N = length(X)
    D = length(X[1])

    out = Matrix{T}(undef,D,N)
    for n = 1:N
        out[:,n] = X[n]
    end

    return out
end

###########  density fit-related

# Boyd's convex optimization convention: log(-2) := Inf.
function evalcvxlog(x::T)::T where T <: Real
    #println("x = ", x)
    if x < zero(T)
        return Inf
    end

    return log(x)
end



function RKHSfitdensitycostfunc(α::Vector{T}, Kp::Matrix{T}, y::Vector{T}, μ::T)::T where T
    N = length(y)
    @assert length(α) == size(Kp,1) == size(Kp,2)

    r = Kp*α-y
    term1 = dot(r,r)
    term2 = μ*dot(α, Kp, α)
    obj = term1 + term2

    return obj
end
# # test code.
# θ = θ_canonical
# y = vec(𝑌)
# μ = σ²
#
# K = constructkernelmatrix(𝑋, θ)
# #α_solution = fitpdfviaSDP(y, K, μ, max_iters)
#
# # verify objective.
# obj = mycostfunc(α_SDP, K, y, μ)
# # this objective scaoe is the same as the cone solver's. 1.69e-01 answer.

########### end of density fit-related

function findboundingbox(X::Vector{Vector{T}}) where T
    D = length(X[1])
    N = length(X)

    limit_a = Vector{T}(undef, D)
    limit_b = Vector{T}(undef, D)

    for d = 1:D
        Xd = collect( X[n][d] for n = 1:N )
        sorted_Xd = sort(Xd)

        limit_a[d] = sorted_Xd[1]
        limit_b[d] = sorted_Xd[end]
    end

    return limit_a, limit_b
end


function bandpasstotalvariations(LP_percentages,
                        LP_attenuations,
                        HP_percentages,
                        HP_attenuations,
                        y::Vector{T}) where T
    #
    N = length(LP_percentages)
    @assert N == length(LP_attenuations) ==
                length(HP_percentages) ==
                length(HP_attenuations)

    #
    tv_array = Vector{T}(undef, N)

    for i = 1:N
        y_bp = SignalTools.applybandpass( LP_percentages[i],
                                LP_attenuations[i],
                                HP_percentages[i],
                                HP_attenuations[i],
                                y )
        #
        tv_array[i] = sum( abs.(y_bp) )
    end

    return tv_array
end

function bandpassmaxmagnitudes(LP_percentages,
                        LP_attenuations,
                        HP_percentages,
                        HP_attenuations,
                        y::Vector{T}) where T
    #
    N = length(LP_percentages)
    @assert N == length(LP_attenuations) ==
                length(HP_percentages) ==
                length(HP_attenuations)

    #
    max_magnitude = Vector{T}(undef, N)

    for i = 1:N
        y_bp = SignalTools.applybandpass( LP_percentages[i],
                                LP_attenuations[i],
                                HP_percentages[i],
                                HP_attenuations[i],
                                y )
        #
        max_magnitude[i] = maximum( abs.(fft(y_bp)) )
    end

    return max_magnitude
end

# 1. FGPGM for parameter identification in systems of nonlinear ODEs,
#       supplementary.
# Let V be the derivative variables of X.
# D_mat is equation 34 of [1]
function evalderivative(x::Vector{T},
                        μ_GP::Vector{T},
                        μ_GP_dot::Vector{T},
                        θ::KT) where {T <: Real, KT}

    #
    K_XX, K_XV, K_VV = constructkernelematrices(X, θ)

    #K_VX = copy(K_XV')
    μ_dot, S_dot = Utilities.getMVNmarginalparams(x, μ_GP_dot, μ_GP,
                                    K_VV, K_XV, K_XX)
    #
    D_mat = K_XV*inv(K_XX)

    return μ_dot, S_dot, D_mat
end


# b splines

# constant (flat) boundary condition.
# see https://github.com/JuliaMath/Interpolations.jl/blob/master/src/b-splines/cubic.jl
function cubicitp1D(c, x::T, a::T, b::T) where T <: Real
    if one(T) <= x <= length(c)

        δx = abs(x - floor(x))
        i = round(Int, x-δx)

        # out = 1/6*c[i-1]*(1+i-δx)^3 + c[i]*(2/3 - (δx-i)^2 + 1/2*(δx-i)^3) +
        #          c[i+1]*(2/3 - (1+i-δx)^2 + 1/2*(1+i-δx)^3) + 1/6*c[i+2]*(δx-i)^3

        out = 1/6*c[i-1]*(1+i-x)^3 + c[i]*(2/3 - (x-i)^2 + 1/2*(x-i)^3) +
                 c[i+1]*(2/3 - (1+i-x)^2 + 1/2*(1+i-x)^3) + 1/6*c[i+2]*(x-i)^3

        return out
    end

    if x < one(T)
        return a
    end

    return b
end

# # ignore boundary conditions for now.
function cubicitp2D(c, x::Vector{T}, N::Vector{Int}, a, b) where T <: Real

    @assert one(T) <= x[1] <= convert(T,N[1])
    @assert one(T) <= x[2] <= convert(T,N[2])


    δx1 = abs(x[1] - floor(x[1]))
    i = round(Int, x[1]-δx1)

    δx2 = abs(x[2] - floor(x[2]))
    j = round(Int, x[2]-δx2)

    running_sum = zero(T)
    for 𝑖 in Iterators.product(i-1:i+2, j-1:j+2)
        𝑖_vec = collect(𝑖)
        running_sum += c[𝑖_vec...]*evalBspline3(x-𝑖_vec)
    end

    return running_sum
end

function evalBspline3(x::Vector{T}) where T <: Real
    return prod( evalBspline3(x[d]) for d = 1:length(x) )
end

function evalBspline3(x::T) where T <: Real
    abs_x = abs(x)

    if zero(T) <= abs_x < one(T)
        return 2/3 - abs_x^2 + 0.5*abs_x^3
    end

    if one(T) <= abs_x < 2.0
        return 1/6 * (2-abs_x)^3
    end

    return zero(T)
end


function cubicitp2Dexplicit(c, x::Vector{T}, N::Vector{Int}) where T <: Real

    @assert one(T) <= x[1] <= convert(T,N[1])
    @assert one(T) <= x[2] <= convert(T,N[2])

    # determine anchor, 𝑖 = (i1,i2)
    δx1 = abs(x[1] - floor(x[1]))
    i1 = round(Int, x[1]-δx1)

    δx2 = abs(x[2] - floor(x[2]))
    i2 = round(Int, x[2]-δx2)

    x1_lag = x[1]-i1
    x2_lag = x[2]-i2

    x1_adv = 1 - x1_lag
    x2_adv = 1 - x2_lag

    out  = c[i1-1, i2-1]*evalBspline3far(x1_lag)*evalBspline3far(x2_lag)
    out += c[i1, i2-1]*evalBspline3near(x1_lag)*evalBspline3far(x2_lag)
    out += c[i1-1, i2]*evalBspline3far(x1_lag)*evalBspline3near(x2_lag)
    out += c[i1, i2]*evalBspline3near(x1_lag)*evalBspline3near(x2_lag)

    out += c[i1-1, i2+1]*evalBspline3far(x1_lag)*evalBspline3near(x2_adv)
    out += c[i1, i2+1]*evalBspline3near(x1_lag)*evalBspline3near(x2_adv)
    out += c[i1-1, i2+2]*evalBspline3far(x1_lag)*evalBspline3far(x2_adv)
    out += c[i1, i2+2]*evalBspline3near(x1_lag)*evalBspline3far(x2_adv)

    out += c[i1+1, i2-1]*evalBspline3near(x1_adv)*evalBspline3far(x2_lag)
    out += c[i1+2, i2-1]*evalBspline3far(x1_adv)*evalBspline3far(x2_lag)
    out += c[i1+1, i2]*evalBspline3near(x1_adv)*evalBspline3near(x2_lag)
    out += c[i1+2, i2]*evalBspline3far(x1_adv)*evalBspline3near(x2_lag)

    out += c[i1+1, i2+1]*evalBspline3near(x1_adv)*evalBspline3near(x2_adv)
    out += c[i1+2, i2+1]*evalBspline3far(x1_adv)*evalBspline3near(x2_adv)
    out += c[i1+1, i2+2]*evalBspline3near(x1_adv)*evalBspline3far(x2_adv)
    out += c[i1+2, i2+2]*evalBspline3far(x1_adv)*evalBspline3far(x2_adv)

    return out
end

function evalBspline3far(x::T) where T <: Real

    return 1/6 * (1-x)^3 # note this isn't (2-abs(x)), since we're using xk_adv.
    # taken from https://github.com/JuliaMath/Interpolations.jl/blob/master/src/b-splines/cubic.jl
end

function evalBspline3near(x::T) where T <: Real
    abs_x = abs(x)

    return 2/3 - x^2 + 0.5*x^3
end


# with respect to the second coordinate.
function cubicitp2Dexplicit2(c, x::Vector{T}, N::Vector{Int}) where T <: Real

    @assert one(T) <= x[1] <= convert(T,N[1])
    @assert one(T) <= x[2] <= convert(T,N[2])

    # determine anchor, 𝑖 = (i1,i2)
    δx1 = abs(x[1] - floor(x[1]))
    i1 = round(Int, x[1]-δx1)

    δx2 = abs(x[2] - floor(x[2]))
    i2 = round(Int, x[2]-δx2)

    x1_lag = x[1]-i1
    x2_lag = x[2]-i2

    x1_adv = 1 - x1_lag
    x2_adv = 1 - x2_lag

    a3 = zero(T)
    a2 = zero(T)
    a1 = zero(T)
    a0 = zero(T)

    k = i2

    ### x2_lag.

    # c[i1-1, i2-1]*evalBspline3far(x1_lag)*evalBspline3far(x2_lag)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3farlag(a3, a2, a1, a0,
                        c[i1-1, i2-1]*evalBspline3far(x1_lag), k)

    # c[i1, i2-1]*evalBspline3near(x1_lag)*evalBspline3far(x2_lag)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3farlag(a3, a2, a1, a0,
                        c[i1, i2-1]*evalBspline3near(x1_lag), k)

    # c[i1-1, i2]*evalBspline3far(x1_lag)*evalBspline3near(x2_lag)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3nearlag(a3, a2, a1, a0,
                        c[i1-1, i2]*evalBspline3far(x1_lag), k)

    # c[i1, i2]*evalBspline3near(x1_lag)*evalBspline3near(x2_lag)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3nearlag(a3, a2, a1, a0,
                        c[i1, i2]*evalBspline3near(x1_lag), k)

    ##
    # c[i1-1, i2+1]*evalBspline3far(x1_lag)*evalBspline3near(x2_adv)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3nearadv(a3, a2, a1, a0,
                        c[i1-1, i2+1]*evalBspline3far(x1_lag), k)

    # c[i1, i2+1]*evalBspline3near(x1_lag)*evalBspline3near(x2_adv)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3nearadv(a3, a2, a1, a0,
                        c[i1, i2+1]*evalBspline3near(x1_lag), k)

    # c[i1-1, i2+2]*evalBspline3far(x1_lag)*evalBspline3far(x2_adv)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3faradv(a3, a2, a1, a0,
                        c[i1-1, i2+2]*evalBspline3far(x1_lag), k)

    # c[i1, i2+2]*evalBspline3near(x1_lag)*evalBspline3far(x2_adv)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3faradv(a3, a2, a1, a0,
                        c[i1, i2+2]*evalBspline3near(x1_lag), k)

    ##
    # c[i1+1, i2-1]*evalBspline3near(x1_adv)*evalBspline3far(x2_lag)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3farlag(a3, a2, a1, a0,
                        c[i1+1, i2-1]*evalBspline3near(x1_adv), k)

    # c[i1+2, i2-1]*evalBspline3far(x1_adv)*evalBspline3far(x2_lag)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3farlag(a3, a2, a1, a0,
                        c[i1+2, i2-1]*evalBspline3far(x1_adv), k)

    # c[i1+1, i2]*evalBspline3near(x1_adv)*evalBspline3near(x2_lag)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3nearlag(a3, a2, a1, a0,
                        c[i1+1, i2]*evalBspline3near(x1_adv), k)

    # c[i1+2, i2]*evalBspline3far(x1_adv)*evalBspline3near(x2_lag)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3nearlag(a3, a2, a1, a0,
                        c[i1+2, i2]*evalBspline3far(x1_adv), k)

    ##
    # c[i1+1, i2+1]*evalBspline3near(x1_adv)*evalBspline3near(x2_adv)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3nearadv(a3, a2, a1, a0,
                        c[i1+1, i2+1]*evalBspline3near(x1_adv), k)

    # c[i1+2, i2+1]*evalBspline3far(x1_adv)*evalBspline3near(x2_adv)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3nearadv(a3, a2, a1, a0,
                        c[i1+2, i2+1]*evalBspline3far(x1_adv), k)

    # c[i1+1, i2+2]*evalBspline3near(x1_adv)*evalBspline3far(x2_adv)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3faradv(a3, a2, a1, a0,
                        c[i1+1, i2+2]*evalBspline3near(x1_adv), k)

    # c[i1+2, i2+2]*evalBspline3far(x1_adv)*evalBspline3far(x2_adv)
    a3, a2, a1, a0 = getmonomialcoefficientsBspline3faradv(a3, a2, a1, a0,
                        c[i1+2, i2+2]*evalBspline3far(x1_adv), k)

    return a3, a2, a1, a0
end

# monomial coefficients for evalBspline3far(adv)
## for β_far(x_k,adv): expand ((x-k)^3)/6
# x^3/6 -k*x^2/2 +k^2*x/2 -k^3/6
function getmonomialcoefficientsBspline3faradv(a3, a2, a1, a0,
                                                B::T,
                                                k::Int)::Tuple{T,T,T,T} where T <: Real
    b3 = 1/6
    b2 = -k/2
    b1 = k^2/2
    b0 = -k^3/6

    return B*b3+a3, B*b2+a2, B*b1+a1, B*b0+a0
end

# monomial coefficients for evalBspline3far(lag)
## for β_far(x_k,lag): expand ((1-(x-k))^3)/6
# k^3/6 - k^2*x/2 + k^2/2 + k*x^2/2 - k*x + k/2 - x^3/6 + x^2/2 - x/2 + 1/6
function getmonomialcoefficientsBspline3farlag(a3, a2, a1, a0,
                                                B::T,
                                                k::Int)::Tuple{T,T,T,T} where T <: Real
    b3 = -1/6
    b2 = (1 + k)/2
    b1 = -(1 + 2*k + k^2)/2
    b0 = (1 + 3*k + 3*k^2 + k^3)/6

    return B*b3+a3, B*b2+a2, B*b1+a1, B*b0+a0
end

# monomial coefficients for evalBspline3near(adv)
## for β_near(x_k,adv): expand 2/3 -(1-(x-k))^2+((1-(x-k))^3)/2
# 2/3 + k^3/2 - 3*k^2*x/2 + k^2/2 + 3*k*x^2/2 -k*x -k/2 -x^3/2 +x^2/2 +x/2 -0.5
function getmonomialcoefficientsBspline3nearadv(a3, a2, a1, a0,
                                                B::T,
                                                k::Int)::Tuple{T,T,T,T} where T <: Real
    b3 = -0.5
    b2 = (1 + 3*k )/2
    b1 = (1 - 2*k - 3*k^2)/2
    b0 = (-1 -k + k^2 + k^3)/2 + 2/3

    return B*b3+a3, B*b2+a2, B*b1+a1, B*b0+a0
end

# monomial coefficients for evalBspline3near(lag)
## for β_near(x_k,lag): expand 2/3 -(x-k)^2 +((x-k)^3)/2
# -k^3/2 + 3*k^2*x/2 -k^2 -3*k*x^2/2 +2*k*x +x^3/2 -x^2 +2/3
function getmonomialcoefficientsBspline3nearlag(a3, a2, a1, a0,
                                                B::T,
                                                k::Int)::Tuple{T,T,T,T} where T <: Real
    b3 = 0.5
    b2 = -1 - 1.5*k
    b1 = 2*k + 1.5*k^2
    b0 = 2/3 - k^2 - k^3/2

    return B*b3+a3, B*b2+a2, B*b1+a1, B*b0+a0
end

### drawing RVs.
function drawcategorical(w::Vector{T})::Int where T <: Real
    x = rand()

    tmp = 0.0
    for k = 1:length(w)-1
        if x < w[k]+tmp
            return k
        end
        tmp += w[k]
    end

    return length(w)
end

### iterated Brownian Bridge

function Bernoullipolynomial0(x::T)::T where T <: Real
    return one(T)
end

function Bernoullipolynomial1(x::T)::T where T <: Real
    return x-0.5
end

function Bernoullipolynomial2(x::T)::T where T <: Real
    return x^2 -x +1/6
end

function Bernoullipolynomial3(x::T)::T where T <: Real
    return x^3 -1.5*x^2 + 0.5*x
end

function Bernoullipolynomial4(x::T)::T where T <: Real
    return x^4 -2*x^3 +x^2 -1/30
end

function Bernoullipolynomial5(x::T)::T where T <: Real
    return x^5 -2.5*x^4 +5/3*x^3 -1/6*x
end

function Bernoullipolynomial6(x::T)::T where T <: Real
    return x^6 -3*x^5 +5/2*x^4 -0.5*x^2 +1/42
end

### visualization helpers.



# options for warp map samples.
# this is the case when the target density is unknown, but realizations are available.
function getwarpmapsamplecustom( y::Array{T,D},
                            ω_set,
                            pass_band_factor) where {T,D}
    #
    N_bands = length(ω_set)

    Y = y

    #### Split-band analysis.
    ϕY, ψY = SignalTools.runsplitbandanalysis(Y, ω_set, SignalTools.getGaussianfilters)
    ηY = SignalTools.runbandpassanalysis(Y, ω_set, pass_band_factor, SignalTools.getGaussianfilters)

    # #### Riesz transform on the different filtered signals.
    # H, ordering = gethigherorderRTfilters(Y,order)
    #
    # 𝓡ϕY = collect( RieszAnalysisLimited(ϕY[s],H) for s = 1:N_bands)
    # 𝓡ψY = collect( RieszAnalysisLimited(ψY[s],H) for s = 1:N_bands)
    # 𝓡ηY = collect( RieszAnalysisLimited(ηY[s],H) for s = 1:N_bands)

    ϕ_set = ηY

    ϕ = reduce(+,ϕ_set)./N_bands

    return ϕY, ψY, ηY
end

function getwarpmaplinear(ϕ::Array{T,D}) where {T,D}

    itp_ϕ = Interpolations.interpolate(ϕ,
                Interpolations.BSpline(Interpolations.Linear()))

    etp_ϕ = Interpolations.extrapolate(itp_ϕ, Interpolations.Line())

    return etp_ϕ
end

# 1-D CDF: integrate from -Inf to b.
# f is the probability density function.
# check the Jacobian: derivative of b-x/(1-x)-0.5.
# check the b bound becomes -1: evaluate b-x/(1-x)-0.5, x = -1
# for use with HCubature, so vector-valued input.
function get1DCDFintegrand(b::T, f::Function) where T <: Real

    return tt->f(b - tt[1]/(1-tt[1]) -0.5)*(1/(1-tt[1])^2)
end

function getKRCDFintegrand(b::T, f::Function) where T <: Real

    return tt->f(b - tt[end]/(1-tt[end]) -0.5)*(1/(1-tt[end])^2) # Sign flip because we have the bounds reversed.
end

# compute the normalizing constant.
# for use with HCubature, so vector-valued input.
function get1DZintegrand(f::Function) where T <: Real

    return tt->f(tt[1]/(1-tt[1]^2))*((1+tt[1]^2)/(1-tt[1]^2)^2)
end

# computes 1D normalizing constant in the last input component of f.
function getKRZintegrand(f::Function) where T <: Real

    return tt->f(tt[end]/(1+tt[end])^2)*(1+tt[end]^2)/(1-tt[end]^2)^2
end

# computes 1D normalizing constant in the last input component of f.
function getKRZintegrandlogspace(f::Function) where T <: Real

    return tt->exp( log(f(tt[end]/(1+tt[end])^2)) + log(1+tt[end]^2) - 2*log(1-tt[end]^2) )
end

# returns positions, weights. Not working.
function gettanhsinhquad(N::Int, m::Int)::Tuple{Vector{Float64},Vector{Float64}}
    half_π = π/2

    j_range = -N:N
    h = 1/2^m

    g = tt->tanh(half_π*sinh(tt))
    dg = tt->half_π *cosh(tt)*sech(half_π*sinh(tt))^2 # derivative of tanh(pi/2*sinh(t))

    x = collect( g(h*j) for j in j_range )
    w = collect( dg(h*j) for j in j_range )

    return x, w
end

# evaluates a tanh-sinh quadature integral.
function computetsq(x, w, f, m)
    N_nodes = length(w)
    h = 1/2^m

    return h*sum( w[i]*f(x[i]) for i = 1:N_nodes )
    #return sum( w[i]*f(x[i]) for i = 1:N_nodes )
end

# coordinates.
# x ∈ prod( [a[d], b[d]] for d = 1:D ).
# same sampling interval for each dimension.
# M[d] is the number of samples in dimension d.
# out ∈ prod( 1:M[d] for d = 1:D )
function convert2itpindex(  x::Vector{T},
                            a::Vector{T},
                            b::Vector{T},
                            M::Vector{Int})::Vector{T} where T <: Real

    D = length(x)
    @assert D == length(a) == length(b)

    out = Vector{T}(undef,D)
    for d = 1:D
        offset = b[d]-a[d]
        u = (x[d]-a[d])/offset # ∈ [0,1]
        i = u*(M[d]-1) # ∈ [0,M-1]
        out[d] = i + 1
    end

    return out
end
# x =[-0.01; 2]
# a = [-0.1; -1 ]
# b = [0.0; 3]
# M = [10; 5]
# p = convert2itpindex([-0.01; 3], a, b, M) # should be [9;5]

# marginalize over ℝ for the dimensions specified in u_dims.
# Uses the change-of-variable here:
# https://github.com/stevengj/cubature/blob/master/README.md#infinite-intervals
# function marginalize(f::Function, u_dims::Vector{Float64}, ϵ = 1e-4)
#
# (1+tt[d]^2)/(1-tt[d]^2)^2)
#
#     d = u[1]
#     integrand_func = tt->f(tt[d]/(1-tt[d]^2)
#     for d in u[1:end]
#         integrand_func = tt->integrand_func(tt[d]/(1-tt[d]^2)
#     end
#
#     x_min = -one(Float64)
#     x_max = one(Float64)
#     (val,err) = HCubature.hcubature(integrand_func, [x_min], [x_max];
#                     norm = LinearAlgebra.norm, rtol = sqrt(eps(Float64)), atol = 0,
#                     maxevals = typemax(Int), initdiv=1)
#
#     return val, err
# end

function iterativesin()
    N = 3
    f_array = Vector{Function}(undef,N)

    f_array[1] = xx->sin(xx)
    for i = 2:N
        f_array[i] = xx->f_array[i-1](sin(xx))
    end

    return f_array[end]
end

# # f is the joint pdf.
# # marginalizes over variables N:D, where D is the total number of variables.
# # assumes the N variables have support ℝ.
# function getintegrandformarginalization(v::Vector{T},
#                                         f::Function,
#                                         N::Int,
#                                         D::Int) where T <: Real
#     @assert 0 < N < D # at least one variable remains.
#     @assert length(v) == D-N
#
#     g = tt->tt/(1-tt^2)
#     u = xx->collect( g(xx[d]) for d = N:D )
#
#     J = tt->( (1+tt^2)/(1-tt^2)^2 )
#
#     integrand_func = xx->f( [v; u(xx)] )*prod( J(xx[d]) for d = N:D )
#
#     return integrand_func
# end
#
# # f is joint pdf.
# # based on numerical integration.
# function evalmarginalpdf(f::Function, v::Vector{Float64}, N::Int, D::Int, max_integral_evals::Int = 1000)
#
#     integrand_func = getintegrandformarginalization(v, f, N, D)
#
#     x_min = collect( -1.0 for n = N:D )
#     x_max = collect( 1.0 for n = N:D )
#     (val,err) = HCubature.hcubature(integrand_func, x_min, x_max;
#                     norm = LinearAlgebra.norm, rtol = sqrt(eps(Float64)), atol = 0,
#                     maxevals = max_integral_evals, initdiv = 1)
#     #
#     return val
# end


# f is the joint pdf.
# marginalizes over variables 1:N. D is the total number of variables.
# assumes the N variables have support ℝ.
function getintegrandformarginalization(v::Vector{T},
                                        f::Function,
                                        N::Int,
                                        D::Int) where T <: Real
    @assert N < D # at least one variable remains.
    @assert length(v) == D-N

    g = tt->tt/(1-tt^2)
    u = xx->collect( g(xx[d]) for d = 1:N )

    J = tt->( (1+tt^2)/(1-tt^2)^2 )

    integrand_func = xx->f( [u(xx); v] )*prod( J(xx[d]) for d = 1:N )
    #integrand_func = xx->f( [u(xx); u(v)] )*prod( J(xx[d]) for d = 1:length(xx) )*prod( J(v[d]) for d = 1:length(v) )

    return integrand_func
end
# f is joint pdf.
# based on numerical integration.
function evalmarginalpdf(f::Function, v::Vector{Float64}, N::Int, D::Int, max_integral_evals::Int = 1000)

    integrand_func = getintegrandformarginalization(v, f, N, D)

    x_min = collect( -1.0 for n = 1:N )
    x_max = collect( 1.0 for n = 1:N )
    (val,err) = HCubature.hcubature(integrand_func, x_min, x_max;
                    norm = LinearAlgebra.norm, rtol = sqrt(eps(Float64)), atol = 0,
                    maxevals = max_integral_evals, initdiv = 1)
    #
    return val
end

# f is the joint pdf.
# marginalizes over the last N variables, where D is the total number of variables.
# assumes the N variables have support ℝ.
function getintegrandformarginalizationrev(v::Vector{T},
                                        f::Function,
                                        N::Int,
                                        D::Int) where T <: Real
    @assert 0 < N < D
    @assert length(v) == D- length((D-N+1):D)

    g = tt->tt/(1-tt^2)
    u = xx->collect( g(xx[d]) for d = 1:length(xx) )

    J = tt->( (1+tt^2)/(1-tt^2)^2 )

    integrand_func = xx->f( [v; u(xx)] )*prod( J(xx[d]) for d = 1:length(xx) )

    return integrand_func
end
# f is joint pdf.
# based on numerical integration.
function evalmarginalpdfrev(f::Function, v::Vector{Float64}, N::Int, D::Int, max_integral_evals::Int = 1000)

    integrand_func = getintegrandformarginalizationrev(v, f, N, D)

    x_min = collect( -1.0 for n = (D-N+1):D )
    x_max = collect( 1.0 for n = (D-N+1):D )
    (val,err) = HCubature.hcubature(integrand_func, x_min, x_max;
                    norm = LinearAlgebra.norm, rtol = sqrt(eps(Float64)), atol = 0,
                    maxevals = max_integral_evals, initdiv = 1)
    #
    return val
end

# f is the joint pdf.
# marginalizes over the first N variables.
# assumes the N variables have support ℝ.
function getintegrandforZ(  f::Function,
                            D::Int) where T <: Real

    g = tt->tt/(1-tt^2)
    u = xx->collect( g(xx[d]) for d = 1:D )

    J = tt->( (1+tt^2)/(1-tt^2)^2 )

    integrand_func = xx->f( u(xx) )*prod( J(xx[d]) for d = 1:D )

    return integrand_func
end

# f is joint pdf.
# based on numerical integration.
function evalZ9(f::Function, D::Int, max_integral_evals::Int = 10000)

    integrand_func = getintegrandforZ(f, D)

    x_min = collect( -1.0 for d = 1:D )
    x_max = collect( 1.0 for d = 1:D )
    (val,err) = HCubature.hcubature(integrand_func, x_min, x_max;
                    norm = LinearAlgebra.norm, rtol = sqrt(eps(Float64)), atol = 0,
                    maxevals = max_integral_evals, initdiv = 1)
    #
    return val, err
end

# Uses the change-of-variable here:
# https://github.com/stevengj/cubature/blob/master/README.md#infinite-intervals
function evalindefiniteintegral(f::Function, max_integral_evals = typemax(Int))

    integrand_func = tt->(f(tt[1]/(1-tt[1]^2))*(1+tt[1]^2)/(1-tt[1]^2)^2)
    x_min = -one(Float64)
    x_max = one(Float64)
    (val,err) = HCubature.hcubature(integrand_func, [x_min], [x_max];
                    norm = LinearAlgebra.norm, rtol = sqrt(eps(Float64)), atol = 0,
                    maxevals = max_integral_evals, initdiv=1)

    return val, err
end

# for p(A | B) as a function of f(A,B).
function setupCDFkernel(hyperparam_A::T,
                        hyperparam_canonical_B::T,
                        ϕ_map::Function)::CPDF1DKernelType{T,Spline34KernelType{T}} where T <: Real

    θ_A = RationalQuadraticKernelType(hyperparam_A)

    θ_canonical = Spline34KernelType(hyperparam_canonical_B) # for adaptive kernel.
    θ_B = AdaptiveKernelType(θ_canonical, xx->ϕ_map(xx))

    return CPDF1DKernelType(θ_A, θ_B)
end
