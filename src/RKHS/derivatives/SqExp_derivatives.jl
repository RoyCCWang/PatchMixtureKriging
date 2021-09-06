
"""
Implements:
#     r = x-z
#
#     factor = -2*a*exp(-a*dot(r,r))
#
#     return r .* factor
"""
function eval∂SqExp!(   out::Vector{T},
                        x::Vector{T},
                        z::Vector{T},
                        a::T) where T
    #
    @assert length(out) == length(x)
    for d = 1:length(x)
        out[d] = x[d]-z[d]
    end

    factor = -2*a*exp(-a*dot(out,out))

    for d = 1:length(out)
        out[d] = out[d] * factor
    end

    return nothing
end

function eval∂SqExp(x::Vector{T},
                z::Vector{T},
                a::T) where T
    #
    out = Vector{T}(undef, length(x))

    eval∂SqExp!(out, x, z, a)

    return out
end

"""
Implements:
#     r = x-z
#
#     return collect( 2*a*(2*a*r[d]^2-1)*exp(-a*dot(r,r)) for d = 1:length(x) )
# Wolfram Alpha: d^2/dx^2 (exp(-a (x^2 + y^2)))
"""
function eval∂2SqExp!(   out::Vector{T},
                        x::Vector{T},
                        z::Vector{T},
                        a::T)::Nothing where T
    #
    @assert length(out) == length(x)
    for d = 1:length(x)
        out[d] = x[d]-z[d]
    end

    factor = 2*a*exp(-a*dot(out,out))

    for d = 1:length(out)
        out[d] = (2*a*out[d]^2-1)  * factor
    end

    return nothing
end

function eval∂2SqExp(x::Vector{T},
                z::Vector{T},
                a::T) where T
    #
    out = Vector{T}(undef, length(x))

    eval∂2SqExp!(out, x, z, a)

    return out
end


function queryd1(x, c, X, a::T)::Vector{T} where T

    out = zeros(T,length(x))

    queryd1!(out, x, c, X, a)

    return out
end

function queryd1!(out::Vector{T}, x, c, X, a::T)::Nothing where T

    N = length(X)
    @assert length(c) == length(X)

    D = length(x)

    # out = zeros(T, D)
    # for n = 1:N
    #     out += c[n] .* eval∂SqExp(x, X[n], a)
    # end

    tmp = Vector{T}(undef, D)

    resize!(out, D)
    fill!(out, zero(T))
    for n = 1:N
        eval∂SqExp!(tmp, x, X[n], a)

        for d = 1:D
            out[d] += c[n]*tmp[d]
        end
    end

    return nothing
end

function queryd2(x, c, X, a::T)::Vector{T} where T

    out = zeros(T,length(x))

    queryd2!(out, x, c, X, a)

    return out
end

function queryd2!(out::Vector{T}, x, c, X, a::T)::Nothing where T
    N = length(X)
    @assert length(c) == length(X)

    D = length(x)

    # out = zeros(T, D)
    # for n = 1:N
    #     out += c[n] .* eval∂2SqExp(x, X[n], a)
    # end

    tmp = Vector{T}(undef, D)

    #out = zeros(T, D)
    resize!(out, D)
    fill!(out, zero(T))
    for n = 1:N
        eval∂2SqExp!(tmp, x, X[n], a)

        for d = 1:D
            out[d] += c[n]*tmp[d]
        end
    end

    return nothing
end

function querySqExp(x,
                    c::Vector{T},
                    X::Vector{Vector{T}},
                    ϵ_sq::T)::T where T
    #
    D = length(x)
    r = Vector{T}(undef,D)

    out = zero(T)
    for n = 1:length(X)
        for d = 1:D
            r[d] = x[d]-X[n][d]
        end

        out += c[n]*exp(-ϵ_sq*dot(r,r))
    end

    return out
end


##### others.
function sum∂2SqExp(x::Vector{T},
                z::Vector{T},
                a::T)::T where T
    #
    r = x-z

    out = zero(T)
    for d = 1:length(x)
        tmp = 2*a*(2*a*r[d]^2-1)

        out += tmp
    end
    #out = sqrt(out)*exp(-a*dot(r,r))
    out = out*exp(-a*dot(r,r))

    return out
end

function sum∂4SqExp(x::Vector{T},
                z::Vector{T},
                a::T)::T where T
    #
    r = x-z
    out = zero(T)
    for d = 1:length(x)
        tmp = 4*a^2*x[d]^4 - 12*a*x[d]^2 + 3

        out += tmp
    end
    #out = sqrt(out)*4*a^2*exp(-a*dot(r,r))
    out = out*4*a^2*exp(-a*dot(r,r))

    return out
end


# does not reset!
function add∂2diagSqExp!(out::Vector{T},
                w::T,
                x::Vector{T},
                z::Vector{T},
                a::T) where T
    #
    @assert length(out) == length(x) == length(z)
    r = x-z
    factor = w*exp(-a*dot(r,r))

    for d = 1:length(x)
        tmp = 2*a*(2*a*r[d]^2-1)

        out[d] += tmp*factor
    end

    return out
end
