# multinomial idea.

# x is the input.
# function evalsquaredRKHS(   x::Vector{T},
#                             θ::SquaredFunctionsKernelType,
#                             X::Vector{Vector{T}},
#                             c::Vector{T},
#                             𝓐)::T where T <: Real
#
#     # get kernel components.
#     N = length(X)
#     @assert length(c) == N
#
#     f_x_components = collect( c[i]*evalkernel(x, X[i], θ.base_params) for i = 1:N)
#
#     # sum.
#     out = @distributed (+) for u in 𝓐
#         Combinatorics.multinomial(u...)*prod(f_x_components.^u)
#     end
#
#     return out
# end

function evalsquaredRKHS(   x::Vector{T},
                            θ::SquaredFunctionsKernelType,
                            X::Vector{Vector{T}},
                            c::Vector{T},
                            𝓐)::T where T <: Real

    # get kernel components.
    N = length(X)
    @assert length(c) == N

    f_x_components = collect( c[i]*evalkernel(x, X[i], θ.base_params) for i = 1:N)

    # sum.
    out = sum( Combinatorics.multinomial(u...)*prod(f_x_components.^u) for u in 𝓐 )

    return out
end

# assume y is strictly positive.
# to query: collect( evalsquaredRKHS(x, θp, η.X, η.c, 𝓐) for x in xq )
function fitsquarefunc( X::Vector{Vector{T}},
                        y::Vector{T},
                        θ::KT,
                        σ²::T) where {T,KT}
    #
    N = length(X)

    # fit suare root of y.
    y_sqrt = sqrt.(y)
    η = RKHSProblemType( zeros(T,N),
                         X,
                         θ,
                         σ²)
    fitRKHS!(η, y_sqrt)


    # prepare multinomial indices.
    θp = SquaredFunctionsKernelType(θ)
    𝓐 = SymTensorTools.getsubscriptarray(2,N)

    return η, 𝓐, θp
end
