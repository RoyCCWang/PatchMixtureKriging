
function evalquery(x::Vector{T}, c::Vector{T}, X::Vector{Vector{T}}, θ::KT)::T where {T,KT}

    return sum( c[n]*evalkernel(x, X[n], θ) for n = 1:length(c) )
end

function evalkernelcentersAD( x::Vector{T2},
                            w_x::Vector{T2},
                            n::Int,
                            X::Vector{Vector{T}},
                            θ::FastAdaptiveKernelType{KT,T,D};
                            zero_tol::T = eps(T)*2 )::T2 where {T,KT,T2,D}

    #q = X[n]

    sum_r_warp_sq = sum( (θ.s[i]*(w_x[i] - θ.w_X[n,i]))^2
                            for i = 1:length(θ.warpfuncs))

    # sb = one(T) - sum(θ.s)
    # @assert one(T) >= sb > zero(T)
    sb = one(T)

    sum_r_sq = sum( (sb*(x[d]-X[n][d]))^2 for d = 1:length(x) )

    τ = sqrt( sum_r_sq + sum_r_warp_sq )

    return evalkernel(τ, θ.canonical_kernel)
end

"""
Assumes w_X is up-to-date.
"""
function evalqueryAD( x::Vector{T2},
                    c::Vector{T},
                    X::Vector{Vector{T}},
                    θ::FastAdaptiveKernelType{KT,T,D})::T2 where {T,KT,T2,D}

    w_x = collect( θ.warpfuncs[i](x) for i = 1:length(θ.warpfuncs) )

    return sum( c[n]*evalkernelcentersAD(x, w_x, n, X, θ) for n = 1:length(c) )
end

function setupGPquery(c::Vector{T}, X, θ::KT, σ²::T)::Function where {T,KT}
    N = length(c)

    k_xq_X_buffer = Vector{T}(undef, N)
    A = constructkernelmatrix(X, θ)

    # add observation model's noise.
    for i = 1:size(A,1)
        A[i,i] += σ²
    end

    fq = xx->evalqueryGP!(k_xq_X_buffer, A, xx, c, X, θ)

    return fq
end

function evalqueryGP!(  k_xq_X_buffer::Vector{T},
                        A::Matrix{T},
                        xq::Vector{T},
                        c::Vector{T},
                        X::Vector{Vector{T}},
                        θ::KT)::Tuple{T,T} where {T,KT}
    N = length(c)
    resize!(k_xq_X_buffer, N)

    out_mean = zero(T)
    for n = 1:N
        k_xq_X_buffer[n] = evalkernel(xq, X[n], θ)

        out_mean += c[n]*k_xq_X_buffer[n]
    end

    var_term1 = evalkernel(xq, xq, θ)
    var_term2 = dot(k_xq_X_buffer, A\k_xq_X_buffer)

    return out_mean, var_term1 - var_term2
end




##### for automatic differentiation.

function evalqueryAD(x::Vector{T2}, c::Vector{T}, X::Vector{Vector{T}}, θ::KT)::T2 where {T,T2,KT}

    return sum( c[n]*evalkernel(X[n], x, θ) for n = 1:length(c) )
end

# type unstable. for numerical integration.
function evalqueryunstable(x, c::Vector{T}, X::Vector{Vector{T}}, θ) where T

    return sum( c[n]*evalkernel(X[n], x, θ) for n = 1:length(c) )
end

##### querying derivatives.

"""
    evalnorm∂2query( x,
                    c::Vector{T},
                    X::Vector{Vector{T}},
                    θ::GaussianKernel1DType)::T

Example:
dfq2_AN = xx->evalnorm∂2query(xx, η.c, η.X, η.θ)
dfq2_ND = xx->sqrt(sum( Calculus.hessian(fq,xx)[i,i]^2 for i = 1:length(xx) ))

x = X[3] .- 0.5
dfq2_AN(x)
dfq2_ND(x)
"""
function evalsum∂2query(x,
                        c::Vector{T},
                        X::Vector{Vector{T}},
                        θ::GaussianKernel1DType)::T where T
    #
    out = zero(T)

    for i = 1:length(X)
        out += c[i]*sum∂2SqExp(x,X[i], θ.ϵ_sq)
    end

    return out
end


"""
This is p_ND = xx->collect( Calculus.hessian(fq,xx)[i,i] for i = 1:length(xx) )
"""
function eval∂2query(x,
                        c::Vector{T},
                        X::Vector{Vector{T}},
                        θ::GaussianKernel1DType) where T
    #
    out = zeros(T, length(x))

    for i = 1:length(X)
        add∂2diagSqExp!(out, c[i], x, X[i], θ.ϵ_sq)
    end

    return out
end

"""
evalsumnorm∂2query!(out::Vector{T},
                        x,
                        c::Vector{T},
                        X::Vector{Vector{T}},
                        θ::GaussianKernel1DType)::T

Example:
buffer = zeros(Float64, 1)
p = xx->evalsumnorm∂2query!(buffer, xx, η.c, η.X, η.θ)
p_ND = xx->norm(collect( Calculus.hessian(fq,xx)[i,i] for i = 1:length(xx) ))
println("AN: p(x) = ", p(x))
println("ND: p(x) = ", p_ND(x))
"""
function evalsumnorm∂2query!(out::Vector{T},
                        x,
                        c::Vector{T},
                        X::Vector{Vector{T}},
                        θ::GaussianKernel1DType)::T where T
    #
    D = length(x)
    N = length(X)
    resize!(out, D)
    fill!(out, zero(T))

    for i = 1:N
        add∂2diagSqExp!(out, c[i], x, X[i], θ.ϵ_sq)
    end

    return norm(out)
end
