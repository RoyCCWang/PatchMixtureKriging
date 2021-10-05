
### implements the coveriance matrices as described in Patchwork Kriging for Large-scale Gaussian Process Regression 
# https://jmlr.org/papers/v19/17-042.html


function setuppatchGP!(X_parts::Vector{Vector{Vector{T}}},
    θ::KT) where {T,KT}

    N_parts = length(X_parts)

    ### K_XX.
    K_XX_set = Vector{Matrix{T}}(undef, N_parts)

    
    for n = 1:N_parts

        K_XX_set[n] = constructkernelmatrix(X_parts[n], θ)
    end

    boundary_labels, bb_positive_list, bb_negative_list,
    Xb_positive_list, Xb_negative_list = setupboundaryquantities(X_parts)

    ### K_bb.

    ### K_Xb.

    ### think about fast serach for queryX and queryb.

    return A
end



function setupboundaryquantities(X_parts::Vector{Vector{Vector{T}}}) where T

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

    ### mark part of the non-zero entries of K_bb.
    bb_positive_list = Vector{Tuple{Int,Int}}(undef,0)
    bb_negative_list = Vector{Tuple{Int,Int}}(undef,0)

    #for j = 2:M
        #for i = 1:j-1
    for m1 = 1:N_boundaries
        for m2 = 1:N_boundaries

            k, l = boundary_labels[m1]
            u, v = boundary_labels[m2]
            
            if (k == u && l != v) || (k != u && l == v)
                
                push!(bb_positive_list, (m1,m2))
            
            elseif (k==v && l != u) || (k!=v && l == u)
                
                push!(bb_negative_list, (m1,m2))
            end

        end
    end
    
    ### get non-zero entries of K_Xb.
    
    Xb_positive_list = Vector{Tuple{Int,Int}}(undef,0)
    Xb_negative_list = Vector{Tuple{Int,Int}}(undef,0)

    # number of training data inputs, i.e., excluding patchwork/boundary inputs.
    N_X = sum( length(X_parts[n]) for n = 1:N_parts )

    for m = 1:N_boundaries
        for j = 1:N_parts

            #part_ind = div(i, N_parts) +1
            k, l = boundary_labels[m]

            if k == j
                push!(Xb_positive_list, (m,j))
            
            elseif l == j
                push!(Xb_positive_list, (m,j))
            end
        end
    end

    return boundary_labels, bb_positive_list, bb_negative_list,
        Xb_positive_list, Xb_negative_list
end
# ####### patchworkGP: kernel matrix contruction.
# boundary_labels, bb_positive_list, bb_negative_list,
# Xb_positive_list, Xb_negative_list = RKHSRegularization.setupboundaryquantities(X_parts)

# pause development since the GP inducing-point styled approx. inference could be very slow.
# instead, work on the mixture of GP idea.