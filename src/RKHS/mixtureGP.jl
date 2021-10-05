# mixture GP idea.

function setupmixtureGP(X_parts::Vector{Vector{Vector{T}}},
    θ::KT) where {T,KT}

    N_parts = length(X_parts)

    ### Set up kernel matrices.
    K_XX_set = Vector{Matrix{T}}(undef, N_parts)

    
    for n = 1:N_parts
        X = X_parts[n]
        
        #K_XX_set[n] = constructkernelmatrix(X, θ)

        # train GPs.
        η_set[n] = RKHSRegularization.AdaptiveRKHSProblemType( zeros(T,length(X)),
                     X,
                     θ,
                     σ²)
        RKHSRegularization.fitRKHS!(η, y)

    end

    # TODO implement fast GP inference that skips warp map.
    # remember how fast warpmap was done.
    # warpmap very desirable for mixtureGP.

    

    ### K_bb.

    ### K_Xb.

    ### think about fast serach for queryX and queryb.

    return A
end