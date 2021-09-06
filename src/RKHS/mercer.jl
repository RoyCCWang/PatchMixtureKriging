
# Set the measure to be α*exp(-α^2*x^2)/sqrt(π), for an α > 0.

function getlogγGaussianmercer(β::T, n::Int)::T where T <: Real
    return 0.5*(log(β) -(n-1)*log(2) -lgamma(n))
end

function setupmercer(θ::GaussianKernel1DType, α::T) where T <: Real
    @assert α > zero(T)
    @assert θ.ϵ_sq > zero(T)

    ϵ = sqrt(θ.ϵ_sq)

    β = (1+(2*ϵ/α)^2)^(0.25)
    δ_sq = α^2/2*(β^2-1)

    return ϵ, β, δ_sq
end

# function gethermitepolynomials(x::T, M::Int)::Vector{T} where T <: Real
#     if M == 0
#         return [one(T)]
#     elseif M == 1
#         return [one(T); 2*x]
#     end
#
#     H = Vector{T}(undef, M+1)
#     H[1] = one(T)
#     H[2] = 2*x
#     for i = 2:M
#         m = i-1
#         H[i+1] = 2*x*H[i] -2*m*H[i-1]
#     end
#
#     return H
# end
# # # test code.
# # x = randn()
# # oracle = [1 2*x 4x^2-2 8*x^3-12*x 16*x^4-48*x^2+12]

function evaleigenfunction( θ::GaussianKernel1DType,
                            α::T,
                            β::T,
                            δ_sq::T,
                            n::Int,
                            x::T) where T <: Real

    return exp( getlogγGaussianmercer(β, n) -δ_sq*x^2 ) *GSL.sf_hermite_phys(n-1, α*β*x)
end

# the measure is μ(x) = α/sqrt(π)*exp(-α^2*x^2).
function evaleigenfunctions(θ::GaussianKernel1DType,
                            N::Int,
                            α::T,
                            x::T) where T <: Real

    ϵ, β, δ_sq = setupmercer(θ, α)
    #H = gethermitepolynomials(α*β*x, N-1)

    # eigenfunction evaluations at x.
    ϕ_evals = Vector{T}(undef,N)
    fill!(ϕ_evals,Inf)
    for n = 1:N
        i = n+1

        #ϕ_evals[n] = exp( getlogγGaussianmercer(β, n) -δ_sq*x^2 ) *GSL.sf_hermite_phys(n-1,α*β*x)
        ϕ_evals[n] = evaleigenfunction(θ, α, β, δ_sq, n, x)
    end

    return ϕ_evals
end

function geteigenvalues(    θ::GaussianKernel1DType,
                            N::Int,
                            α::T) where T <: Real

    ϵ, β, δ_sq = setupmercer(θ, α)
    α_sq = α^2
    ϵ_sq = ϵ^2

    # eigenfunction evaluations at x.
    λn = Vector{T}(undef,N)
    fill!(λn,Inf)
    for n = 1:N
        λn[n] = sqrt(α_sq/(α_sq+δ_sq+ϵ_sq))*(ϵ_sq/(α_sq+δ_sq+ϵ_sq))^(n-1)
    end

    return λn
end


# explore SOS
# M is the truncation limit.
function getQmat(c::Vector{T}, ϕ::Function, X, M::Int) where T <: Real
    Q = Matrix{T}(undef,M,M)

    for i = 1:M
        for j = 1:M
            Q[i,j] = sum( c[l]*ϕ(i,X[l])*ϕ(j,X[l]) for l = 1:length(c) )
        end
    end

    return Q
end

# get the Φ matrix.
function getΦmat(ϕ::Function, X::Vector{Vector{T}}, M::Int) where T <: Real
    N = length(X)

    Φ = Matrix{T}(undef, M, N)
    for m = 1:M
        for i = 1:N
            Φ[m,i] = ϕ(m,X[i])
        end
    end

    return Φ
end

# get the Ψ matrix. ψ_b = ϕ_i * ϕ_j, b is col-major ordering.
function getΨmat(ϕ::Function, X::Vector{Vector{T}}, M::Int) where T <: Real
    N = length(X)
    𝑖 = CartesianIndices((M,M))

    Ψ = Matrix{T}(undef, M*M, N)
    fill!(Ψ, Inf)
    for m = 1:M*M
        for l = 1:N

            Ψ[m,l] = ϕ(𝑖[m][1],X[l])*ϕ(𝑖[m][2],X[l])
        end
    end

    return Ψ
end

# alternative to getΨmat. Does the same thing. Less computationally efficient.
function getΨmat2(ϕ::Function, X::Vector{Vector{T}}, M::Int) where T <: Real
    N = length(X)
    Φ = getΦmat(ϕ, X, M)
    B = kron(Φ,Φ)

    Ψ = Matrix{T}(undef, M*M, N)
    for l = 1:N
        Ψ[:,l] = B[:,(l-1)*N+l]
    end

    return Ψ
end

# alternative to getΨmat. Does the same thing. Less computationally efficient.
function getQfromα(ϕ::Function, X::Vector{Vector{T}}, M::Int, α::Vector{T}) where T <: Real

    Q = zeros(T, M, M)
    for i = 1:M
        for j = 1:M
            for l = 1:N
                Q[i,j] += α[l]*ϕ(i,X[l])*ϕ(j,X[l])
            end
        end
    end

    return Q
end

function getQvec(Q::Matrix{T}, M::Int) where T <: Real
    N = length(X)
    𝑖 = CartesianIndices((M,M))

    Q_vec = Vector{T}(undef, M*M)
    for m = 1:M*M
        Q_vec[m] = Q[𝑖[m]]
    end

    return Q_vec
end


function generatecandQSOS(α::Vector{T}, ϕ::Function, X::Vector{Vector{T}}) where T <: Real
    N = length(X)

    Q = Matrix{T}(undef, M, M)
    for i = 1:M
        for j = 1:M
            Q[i,j] = sum( α[l]*ϕ(i,X[l])*ϕ(j,X[l]) for l = 1:N )
        end
    end

    return Q
end


function evalChebyshevpolynomial(n::Int, x::T)::T where T <: Real
    return cos(n*acos(x))
end
