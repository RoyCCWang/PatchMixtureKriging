
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
    Xb_positive_list, Xb_negative_list = setupboundaryXobjects(X_parts)

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

    ### mark part of the non-zero entries of K_bb.
    bb_positive_list = Vector{Tuple{Int,Int}}(undef,0)
    bb_negative_list = Vector{Tuple{Int,Int}}(undef,0)

    for j = 2:M
        for i = 1:j-1
            k, l = boundary_labels[i]
            u, v = boundary_labels[j]
            
            if (k == u && l != v) || (k != u && l == v)
                
                push!(bb_positive_list, (i,j))
            
            elseif (k==v && l != u) || (k!=v && l == v)
                
                push!(bb_negative_list, (i,j))
            end
        end
    end
    
    ### get non-zero entries of K_Xb.
    
    Xb_positive_list = Vector{Tuple{Int,Int}}(undef,0)
    Xb_negative_list = Vector{Tuple{Int,Int}}(undef,0)

    # number of training data inputs, i.e., excluding patchwork/boundary inputs.
    N_X = sum( length(X_parts[n]) for n = 1:length(X_parts) )

    for j = 1:length(boundary_labels)
        for i = 1:N_X

            part_ind = div(i, N_parts) +1
            k, l = boundary_labels[j]

            if k == part_ind
                push!(Xb_positive_list, (i,j))
            
            elseif l == part_ind
                push!(Xb_positive_list, (i,j))
            end
        end
    end

    return boundary_labels, bb_positive_list, bb_negative_list,
        Xb_positive_list, Xb_negative_list
end


function myfunc()
    #

end