# Methods for fitting and querying under the RKHS regularization framework.

# for univariate inputs.
function constructkernelmatrix( X,
                                θ)::Matrix{Float64} where T
    #
    K = Matrix{Float64}(undef,length(X),length(X))
    constructkernelmatrix!(K,X,θ)

    return K
end

function constructkernelmatrix!(K::Matrix{T},
                                X,
                                θ)::Matrix{T} where T

    M = length(X)
    @assert size(K) == (M,M)

    fill!(K,Inf) # debug
    for j = 1:M
        for i = j:M
            K[i,j] = evalkernel(X[i], X[j], θ)
        end
    end

    for j = 2:M
        for i = 1:(j-1)
            K[i,j] = K[j,i]
        end
    end

    return K
end


# experimental.
function constructkernelmatrix2(X::Vector{Vector{T}}, θ_array::Vector)::Matrix{T} where T
    M = length(X)
    @assert length(θ_array) == M

    K = Matrix{T}(undef,M,M)
    fill!(K,Inf) # debug
    for j = 1:M
        #for i = j:M
        for i = 1:M
            K[i,j] = evalkernel(X[i], X[j], θ_array[j])
        end
    end

    # for j = 2:M
    #     for i = 1:(j-1)
    #         K[i,j] = K[j,i]
    #     end
    # end
    # K = (K+K') ./ 2
    #
    # if !isposdef(K)
    #     s,Q = eigen(K)
    #     println("K not posdef! eigenvalues are:")
    #     display(sort(s))
    # end

    return K
end

# v are the derivatives of x.
function constructkernelematrices( X::Vector{T},
                                   θ::ODEKernelType) where T
    N_x = length(X)
    N_v = length(X)

    K = constructkernelmatrix(X, θ)

    K_xv = Matrix{T}(undef, N_x, N_v)
    fill!(K_xv,Inf) # debug
    for j = 1:N_v
        for i = 1:N_x
            K_xv[i,j] = evalkernelderivative(X[i], X[j], θ)
        end
    end

    K_vv = Matrix{T}(undef, N_x, N_v)
    fill!(K_vv,Inf) # debug
    for j = 1:N_v
        for i = 1:N_x
            K_vv[i,j] = evalDkernel𝐷(X[i], X[j], θ)
        end
    end

    return K, K_xv, K_vv
end

# returns K_XZ. Entry i,j is k(x[i],z[j]).
function constructkernelmatrix( X::Vector{Vector{T}},
                                Z::Vector{Vector{T}},
                                θ)::Matrix{T} where T
    Nr = length(X)
    Nc = length(Z)

    K = Matrix{T}(undef,Nr,Nc)
    fill!(K,Inf) # debug
    for j = 1:Nc
        for i = 1:Nr
            K[i,j] = evalkernel(X[i], Z[j], θ)
        end
    end

    return K
end


function constructkernelmatrix!(K,
            η::RKHSProblemType{KT,T,XT}) where {KT,T,XT}

    #
    constructkernelmatrix!(K, η.X, η.θ)

    return nothing
end

function constructkernelmatrix!(K,
            η::RKHSProblemType{FastAdaptiveKernelType{KT,T,D},T,XT}) where {KT,T,XT,D}

    constructkernelmatrix!(K, η.X, η.θ)

    return nothing
end

# for multivariate inputs.
# for use with adaptive kernels.
function constructkernelmatrix!(    K::Matrix{T},
                                    X::Vector{Vector{T}},
                                    θ::FastAdaptiveKernelType{KT,T,D}) where {KT,T,D}
    M = length(X)
    @assert size(K) == (M,M)

    # update warpmap evals.
    for i = 1:length(θ.warpfuncs)
        for n = 1:length(X)
            θ.w_X[n,i] = θ.warpfuncs[i](X[n])
        end
    end

    #sb = one(T) - sum(θ.s)
    #@assert one(T) >= sb > zero(T)
    sb = one(T)

    fill!(K,Inf) # debug
    for j = 1:M
        for i = j:M
            if i == j
                K[i,j] = one(T)
            else
                K[i,j] = evalkernel(X[i], X[j], θ)
            end
        end
    end

    for j = 2:M
        for i = 1:(j-1)
            K[i,j] = K[j,i]
        end
    end

    return nothing
end

function constructkernelmatrix!( X::Vector{Vector{T}},
                                θ::FastAdaptiveKernelType)::Matrix{T} where T
    #
    K = Matrix{T}(undef,length(X),length(X))
    constructkernelmatrix!(K, X, θ)

    return K
end

#### Try separable RKHS via inverse. Assume N_r == D
# Assume η.a is same for all dimensions.


function fitRKHS!(  η,
                    y::Vector{T}) where T
    #
    N = length(η.X)
    K = Matrix{T}(undef, N, N)
    fitRKHS!(K, η, y)

    return nothing
end

"""
U is K + σ²I.
"""
function fitRKHS!(  U::Matrix{T},
                    η,
                    y::Vector{T}) where T

    @assert !isempty(η.X)
    @assert !isempty(y)

    M = length(η.X)
    @assert M == length(y)

    # update kernel matrix
    constructkernelmatrix!(U, η.X, η.θ)

    # add observation model's noise.
    for i = 1:size(U,1)
        U[i,i] += η.σ²[1]
    end

    # solve.
    η.c[:] = U\y

    return nothing
end

# Version where Xq being SharedArray, ! versoin.
function query!(    Yq::Vector{T},
                    Xq::Vector{Vector{T}},
                    η::RKHSProblemType{KT,T}) where {KT, T}

    N = length(η.X)
    @assert !isempty(Xq)

    @assert size(Yq) == size(Xq)

    # Pre-allocate
    kq = Vector{T}(undef,N)
    Nq = length(Xq)

    for iq = 1:Nq

        for i=1:N
            kq[i] = evalkernel(Xq[iq], η.X[i], η.θ)
        end

        Yq[iq] = dot(kq,η.c)

        # # variance.
        # v = L\kq
        # vq[iq] = evalkernel(Xq[iq], Xq[iq], η.θ) - dot(v,v)
    end

    return nothing
end

function query!(    Yq::Vector{T},
                    Xq::Vector{T},
                    η::RKHSProblemType{KT,T}) where {KT, T}

    N = length(η.X)
    @assert !isempty(Xq)

    @assert size(Yq) == size(Xq)

    # Pre-allocate
    kq = Vector{T}(undef,N)
    Nq = length(Xq)

    for iq = 1:Nq

        for i=1:N
            kq[i] = evalkernel(Xq[iq], η.X[i], η.θ)
        end

        Yq[iq] = dot(kq,η.c)

        # # variance.
        # v = L\kq
        # vq[iq] = evalkernel(Xq[iq], Xq[iq], η.θ) - dot(v,v)
    end

    return nothing
end

function query!(    Yq::Vector{T},
                    Xq::Vector{Vector{T}},
                    η::RKHSProblemType{Vector{KT},T}) where {KT, T}

    N = length(η.X)
    @assert !isempty(Xq)

    @assert size(Yq) == size(Xq)

    # Pre-allocate
    kq = Vector{T}(undef,N)
    Nq = length(Xq)

    for iq = 1:Nq

        for i=1:N
            kq[i] = evalkernel(Xq[iq], η.X[i], η.θ[i])
        end

        Yq[iq] = dot(kq,η.c)

        # # variance.
        # v = L\kq
        # vq[iq] = evalkernel(Xq[iq], Xq[iq], η.θ) - dot(v,v)
    end

    return nothing
end



#### posdef checks.
function zerobarrier(x::T, a::T)::T where T <: Real

    #return one(T)/(a*x)^2
    return exp(-(x-a))
    #return abs(x)*a
end

function zerobarrier(x::Complex{T}, a::T)::T where T <: Real

    #return one(T)/abs2(a*x)
    return exp(-(abs(x)-a))
    #return abs(x)*a
end

function zerobarrier(x::Vector{T2}, a::T)::T where {T <: Real, T2}

    #return maximum( zerobarrier(x[d], a) for d = 1:length(x) )
    return sum( zerobarrier(x[d], a) for d = 1:length(x) )/length(x)
end

# function piecewisezerobarrier(x::T, a::T, b::T)::T where T <: Real
#     @assert b > a >= 0.0
#
#     c = (b-1)/a
#
#     if a < x < b
#         return -log(b-c*x)
#     end
#
#     if x > b
#         return Inf
#     end
#
#     return zero(T)
# end

# x must be positive, a is the threshold for 0.0.
function piecewisezerobarrier(x::T, a::T)::T where T <: Real
    @assert x > zero(T)

    if a < x
        return exp(x-a) - one(T)
    end

    return zero(T)
end


function logisticfunc(L, k, x0::T, x)::T where T <: Real

    return L/(1+exp(-k*(x-x0)))
end

"""
y is data here.
"""
function discrepancyscorefromdata(  K::Matrix{T},
                                     y::Vector{T},
                                     σ²::T,
                                     c::Vector{T};
                                     a_cond::T = 1e-10) where T


   discrepancy = norm(( K + σ² .* LinearAlgebra.I )*c - y)

   return exp(discrepancy*a_cond)-one(T)
end

function discrepancyscorefromdata(  U::Matrix{T},
                                     y::Vector{T},
                                     c::Vector{T};
                                     a_cond::T = 1e-10) where T


   discrepancy = norm(U*c - y)

   return exp(discrepancy*a_cond)-one(T)
end


"""
The lower the score, the more well-conditioned K is.
"""
function evalconditioningscore( K::Matrix{T},
                                display_flag::Bool,
                                a_posdef_chk::T,
                                y::Vector{T},
                                σ²::T,
                                c::Vector{T})::T where T

    #s, Q_unused = eigen(K)
    #score = zerobarrier(s, a_posdef_chk)

    # score = piecewisezerobarrier(   cond(K),
    #                                 a_posdef_chk/eps(T))
    score = discrepancyscorefromdata(K, y, σ², c; a_cond = a_posdef_chk)


    if display_flag
        if !isposdef(K)
            println("K not posdef! eigenvalues are:")
            s,Q = eigen(K)
            display(sort(s))
        end
    end

    return score
end

function evalconditioningscore( U::Matrix{T},
                                display_flag::Bool,
                                a_posdef_chk::T,
                                y::Vector{T},
                                c::Vector{T})::T where T

    score = discrepancyscorefromdata(U, y, c; a_cond = a_posdef_chk)


    if display_flag
        if !isposdef(U)
            println("U not posdef! eigenvalues are:")
            s,Q = eigen(U)
            display(sort(s))
        end
    end

    return score
end
