# mixture GP idea.
# go with a new approach: non-object-oriented usage for mixtureGP: don't include θ in η.

function setupmixtureGP(X_parts::Vector{Vector{Vector{T}}},
    y_parts::Vector{Vector{T}},
    θ::KT, σ²) where {T,KT}

    N_parts = length(X_parts)

    # get boundary labels.
    M = round(Int, N_parts*(N_parts-1)/2)
    boundary_labels = Vector{Tuple{Int,Int}}(undef, M)
    a = 0

    for j = 2:N_parts
        for i = 1:j-1
            a += 1
            boundary_labels[a] = (i,j)
        end
    end
    N_boundaries = length(boundary_labels)

    ### Set up kernel matrices.
    K_XX_set = Vector{Matrix{T}}(undef, N_parts)

    η = MixtureGPType(X_parts, boundary_labels)

    for n = 1:N_parts

        K = constructkernelmatrix(X_parts[n], θ)

        # train GPs.
        #B = AdaptiveRKHSProblemType( zeros(T,0), X, θ, σ²)
        #fitRKHS!(K, B, y)
        
        fitRKHS!(K, θ, y_parts[n])

        # store.
        η.K_set[n] = K
        η.c_set[n] = c
        η.σ²_set[n] = σ²
    end

    return η
end

function querymixtureGP(xq::Vector{T},
    K_set::Vector{Matrix{T}},
    X_parts::Vector{Vector{Vector{T}}}, θ::KT, σ²) where {KT, T}

    # get active partition list and weights.

    return nothing
end

function findneighbourpartitions()

end