
### mixture GP types.

# single kernel for all partitions.
mutable struct MixtureGPType{T}
    #
    X_parts::Vector{Vector{Vector{T}}}
    c_set::Vector{Vector{T}}

    # can probably delte the following.
    σ²_set::Vector{T}
    U_set::Vector{Matrix{T}}

    #boundary_labels::Vector{Tuple{Int,Int}}
    hps::Vector{HyperplaneType{T}}

    #MixtureGPType{KT,T}(X_parts::Vector{Vector{Vector{T}}}, θ::KT) where {KT,T} = new{KT,T}(X_parts, θ)
end

function MixtureGPType(X_parts::Vector{Vector{Vector{T}}},
    hps::Vector{HyperplaneType{T}}) where T
    
    N = length(X_parts)

    c_set = Vector{Vector{T}}(undef, N)
    σ²_set = Vector{T}(undef, N)
    K_set = Vector{Matrix{T}}(undef, N)
    
    return MixtureGPType(X_parts, c_set, σ²_set, K_set, hps)
end



function fitmixtureGP!(η::MixtureGPType{T},
    y_parts::Vector{Vector{T}},
    θ::KT, σ²) where {T,KT}

    N_parts = length(η.X_parts)

    ## get boundary labels.
    #M = round(Int, N_parts*(N_parts-1)/2)
    # boundary_labels = Vector{Tuple{Int,Int}}(undef, M)
    # a = 0

    # for j = 2:N_parts
    #     for i = 1:j-1
    #         a += 1
    #         boundary_labels[a] = (i,j)
    #     end
    # end
    # N_boundaries = length(boundary_labels)

    ### Set up kernel matrices.
    #K_XX_set = Vector{Matrix{T}}(undef, N_parts)

    for n = 1:N_parts

        X = η.X_parts[n]
        y = y_parts[n]

        # train GPs.
        U = constructkernelmatrix(X, θ)

        # add observation model's noise.
        for i = 1:size(U,1)
            U[i,i] += σ²
        end

        c = U\y

        # store.
        η.U_set[n] = U
        η.c_set[n] = c
        η.σ²_set[n] = σ²
    end

    return η
end

function querymixtureGP!(Yq::Vector{T},
    Vq::Vector{T},
    Xq::Vector{Vector{T}},
    η::MixtureGPType{T},
    root,
    levels,
    radius::T, 
    δ::T, # tolerance for partition search.
    θ::KT, σ²,
    weight_θ) where {KT, T}

    c_set = η.c_set
    X_parts = η.X_parts
    hps = η.hps

    # set up output buffers.
    Nq = length(Xq)
    resize!(Yq, Nq)
    resize!(Vq, Nq)

    # intermediates.
    kq = Vector{T}(undef, 0)
    w = Vector{T}(undef, 0) # mixture weights for GPs.
    u = Vector{T}(undef, 0) # means of GPs.
    v = Vector{T}(undef, 0) # variances of GPs..

    for j = 1:Nq

        xq = Xq[j]

        # find the region for p.
        p_region_ind = RKHSRegularization.findpartition(xq, root, levels)

        # get active partition list, then use it to get u, v, w.
        region_inds, ts, zs, hps_keep_flags = findneighbourpartitions(xq, radius, root,
            levels, hps, p_region_ind; δ = δ)
        t_kept = ts[hps_keep_flags]

        N_regions = length(region_inds)
        resize!(w, N_regions+1)
        resize!(u, N_regions+1)
        resize!(v, N_regions+1)

        for r = 1:N_regions
            
            w[r] = RKHSRegularization.evalkernel(abs(t_kept[r]), weight_θ)

            u[r], v[r] = queryinner!(kq, xq, X_parts[r], θ, c_set[r])
        end

        # u, v, w for the region that contains xq.
        w[end] = one(T)

        u[end], v[end] = queryinner!(kq, xq, X_parts[p_region_ind], θ, c_set[p_region_ind])

        # normalize to get a convex combination.
        sum_w = sum(w)
        for i = 1:length(w)
            w[i] = w[i]/sum_w
        end

        ## apply mixture.
        Yq[j] = dot(w, u)
        
        #Vq[j] = dot(w,diagm(v)*w)
        Vq[j] = dot(w, v .* w) # faster.
    end

    return nothing
end

function queryinner!(kq::Vector{T}, xq, X, θ, c) where T

    @assert length(c) == length(X)

    resize!(kq, length(X))

    for i = 1:length(X)
        kq[i] = evalkernel(xq, X[i], θ)
    end
    
    # mean.
    μq = dot(kq, c)

    ## variance.
    # v = L\kq
    # vq = evalkernel(Xq[iq], Xq[iq], η.θ) - dot(v,v)
    vq = -1.23 # TODO placeholder for now.

    return μq, vq
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
    ts = Vector{T}(undef, length(hps)) # for debug.

    for i = 1:length(hps)

        # parse.
        u = hps[i].v
        c = hps[i].c

        # intersection with hyperplane.
        t = -dot(u, p) + c
        z = p + t .* u

        zs[i] = z # debug.
        ts[i] = t

        if norm(z-p) < radius
            
            # get two points along ray normal to the plane, on either side of plane.
            z1 = p + (t+δ) .* u
            z2 = p + (t-δ) .* u

            # get z1's, z2's region.
            z1_region_ind = RKHSRegularization.findpartition(z1, root, levels)
            z2_region_ind = RKHSRegularization.findpartition(z2, root, levels)

            # println("i = ", i)
            # println("z1 = ", z1)
            # println("z2 = ", z2)
            # println("z1_region_ind = ", z1_region_ind)
            # println("z2_region_ind = ", z2_region_ind)
            # println()

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

    return region_inds, ts, zs, keep_flags
end