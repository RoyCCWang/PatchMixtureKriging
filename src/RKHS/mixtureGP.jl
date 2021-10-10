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
    #K_XX_set = Vector{Matrix{T}}(undef, N_parts)

    η = MixtureGPType(X_parts, boundary_labels)

    for n = 1:N_parts

        X = X_parts[n]
        y = y_parts[n]

        # train GPs.
        K = constructkernelmatrix(X, θ)
        c = K\y

        # store.
        η.K_set[n] = K
        η.c_set[n] = c
        η.σ²_set[n] = σ²

    end

    return η
end

function querymixtureGP!(Yq::Vector{T},
    Xq::Vector{Vector{T}},
    K_set::Vector{Matrix{T}},
    X_parts::Vector{Vector{Vector{T}}}, θ::KT, σ²) where {KT, T}

    


    # Pre-allocate
    kq = Vector{T}(undef,N)
    Nq = length(Xq)

    for j = 1:Nq

        xq = Xq[j]

        # get active partition list and weights.

        for i=1:N
            kq[i] = evalkernel(xq, η.X[i], η.θ)
        end

        Yq[j] = dot(kq, η.c)

        # # variance.
        # v = L\kq
        # vq[iq] = evalkernel(Xq[iq], Xq[iq], η.θ) - dot(v,v)
    end

    return nothing
end

function fetchhyperplanes(root::BinaryNode{PartitionDataType{T}}) where T

    hps = Vector{HyperplaneType{T}}(undef, 0)
    for node in AbstractTrees.PreOrderDFS(root)
        
        if isdefined(node.data.hp, :v)
            # non-leaf node.
            push!(hps, node.data.hp)
        end
    end
 
    return hps
end

"""
Given a point p and a list of hyperplanes hps, find the regions within adjacent to p within radius.
"""
function findneighbourpartitions(p::Vector{T},
    radius::T,
    root,
    levels,
    hps,
    p_region_ind::Int;
    δ::T = 1e-10) where T
    
    region_inds = Vector{Int}(undef, length(hps))
    j = 0

    keep_flags = falses(length(hps)) # for debug.
    zs = Vector{Vector{T}}(undef, length(hps)) # for debug.

    for i = 1:length(hps)

        # parse.
        u = hps[i].v
        c = hps[i].c

        # intersection with hyperplane.
        t = -dot(u, p) + c
        z = p + t .* u

        zs[i] = z # debug.

        if norm(z-p) < radius
            
            # get two points along ray normal to the plane, on either side of plane.
            z1 = p + (t+δ) .* u
            z2 = p + (t-δ) .* u

            # get z1's, z2's region.
            z1_region_ind = RKHSRegularization.findpartition(z1, root, levels)
            z2_region_ind = RKHSRegularization.findpartition(z2, root, levels)

            println("i = ", i)
            println("z1 = ", z1)
            println("z2 = ", z2)
            println("z1_region_ind = ", z1_region_ind)
            println("z2_region_ind = ", z2_region_ind)
            println()

            # do not keep if both are in p's region.
            # keep if either z1 or z2 is in p's region.
            #if !(z2_region_ind == p_region_ind && z1_region_ind == p_region_ind) && 
            #    (z2_region_ind == p_region_ind || z1_region_ind == p_region_ind)
            if xor(z2_region_ind == p_region_ind, z1_region_ind == p_region_ind)
                
                keep_flags[i] = true

                j += 1
                region_inds[j] = z1_region_ind
                
            end
        end
    end

    resize!(region_inds, j)

    return region_inds, zs, keep_flags
end