

# This is the canonical kernel: 1D Gaussian.
function evalkernel(x, z::T, θ::GaussianKernelODEType)::T where T <: Real

    # a is ϵ^2 in "Stable eval of Gaussian radial basis func interpolants", 2012.
    out::T = θ.s[1]*exp(-θ.ϵ_sq[1]*(x-z)^2)

    @assert isfinite(out)
    return out
end

"""
evalDkernel(x, z::T, θ)::T

evaluates ∂k(a,b)_∂a at (a,b) = (x,z). See Mathematica script for derivation.
"""
function evalDkernel(x, z::T, θ::GaussianKernelODEType)::T where T <: Real

    τ::T = x-z
    out::T = -2*θ.ϵ_sq[1]*τ*exp(-θ.ϵ_sq[1]*τ^2)
    out = θ.s[1]*out

    @assert isfinite(out)
    return out
end

"""
evalkernel𝐷(x, z::T, θ)::T

evaluates ∂k(a,b)_∂b at (a,b) = (x,z).
See Mathematica script for derivation.

Test:

import PatchMixtureKriging
using LinearAlgebra
θ = PatchMixtureKriging.GaussianKernelODEType(1.23, 0.321)
s = randn()
z = randn()
p1 = PatchMixtureKriging.evalDkernel(s,z,θ)
p2 = PatchMixtureKriging.evalkernel𝐷(z,s,θ)
println("discrepancy = ", norm(p1-p2))

"""
function evalkernel𝐷(x, z::T, θ::GaussianKernelODEType)::T where T <: Real

    τ::T = x-z
    out::T = 2*θ.ϵ_sq[1]*τ*exp(-θ.ϵ_sq[1]*τ^2)
    out = θ.s[1]*out

    @assert isfinite(out)
    return out
end

"""
evalDkernel𝐷(x, z::T, θ)::T

evaluates ∂k(a,b)_∂a_∂b at (a,b) = (x,z). See Mathematica script for derivation.
"""
function evalDkernel𝐷(x, z::T, θ::GaussianKernelODEType)::T where T <: Real

    τ_sq::T = (x-z)^2
    out::T = 2*θ.ϵ_sq[1] *exp(-θ.ϵ_sq[1] *τ_sq) *(1 - 2 *θ.ϵ_sq[1] *τ_sq)
    out = θ.s[1]*out

    @assert isfinite(out)
    return out
end



##########
