

mutable struct HyperplaneType{T}
    v::Vector{T}
    c::T
    
    HyperplaneType{T}(v,c) where T = new{T}(v,c)
    HyperplaneType{T}() where T = new{T}()
end

mutable struct PartitionDataType{T}
    hp::HyperplaneType{T}
    X::Vector{Vector{T}}
    global_X_indices::Vector{Int}
    index::Int
end

mutable struct BinaryNode{T}
    data::T
    parent::BinaryNode{T}
    left::BinaryNode{T}
    right::BinaryNode{T}

    # Root constructor
    BinaryNode{T}(data) where T = new{T}(data)
    # Child node constructor
    BinaryNode{T}(data, parent::BinaryNode{T}) where T = new{T}(data, parent)
end
BinaryNode(data) = BinaryNode{typeof(data)}(data)


"""
Mutates parent. Taken from AbstractTrees.jl's example code.
"""
function leftchild!(parent::BinaryNode, data)
    !isdefined(parent, :left) || error("left child is already assigned")
    node = typeof(parent)(data, parent)
    parent.left = node
end

"""
Mutates parent. Taken from AbstractTrees.jl's example code.
"""
function rightchild!(parent::BinaryNode, data)
    !isdefined(parent, :right) || error("right child is already assigned")
    node = typeof(parent)(data, parent)
    parent.right = node
end

"""
Taken from AbstractTrees.jl's example code.
"""
function AbstractTrees.children(node::BinaryNode)
    if isdefined(node, :left)
        if isdefined(node, :right)
            return (node.left, node.right)
        end
        return (node.left,)
    end
    isdefined(node, :right) && return (node.right,)
    return ()
end

function splitpoints(u::Vector{T}, X::Vector{Vector{T}}) where T <: Real
    
    N = length(X)
    indicators = falses(N)

    functional_evals = collect( dot(u, X[n]) for n = 1:N )
    c = Statistics.median(functional_evals)

    for n = 1:N
        
        if functional_evals[n] < c

            indicators[n] = true
        else
            indicators[n] = false
        end
    end

    return indicators, functional_evals, c
end 


function gethyperplane(X::Vector{Vector{T}}) where T
    
    # center.
    μ = Statistics.mean(X)
    Z = collect( X[n] - μ for n = 1:size(X,2) )

    Z_mat = (array2matrix(Z))'
    U, s, V = svd(Z_mat)
    v = V[:,1]

    indicators, functional_evals, c = splitpoints(v, X)
    hp = HyperplaneType{T}(v, c)

    return hp, indicators
end


"""
current node is p.
"""
function setuppartition(X::Vector{Vector{T}}, level) where T

    # get hyperplane.
    hp, left_indicators = gethyperplane(X)
    
    X_empty = Vector{Vector{T}}(undef, 0)
    #data = PartitionDataType(hp, X_empty, 0)

    X_empty_inds = Vector{Int}(undef, 0)
    data = PartitionDataType(hp, X_empty, X_empty_inds, 0)

    # add to current node.
    root = BinaryNode(data)

    # might have to use recursion.
    X_inds = collect(1:length(X))
    createchildren(root, left_indicators, "left", X, X_inds, level-1)
    createchildren(root, left_indicators, "right", X, X_inds, level-1)
    
    # assign consecutive whole numbers as indices for the leaf nodes.
    X_parts, X_parts_inds = labelleafnodes(root, X)

    return root, X_parts, X_parts_inds
end

function labelleafnodes(root::BinaryNode{PartitionDataType{T}},
    X::Vector{Vector{T}}) where T
    #
    
    X_parts = Vector{Vector{Vector{Float64}}}(undef, 0)
    X_parts_inds = Vector{Vector{Int}}(undef, 0)
    leaves = Vector{BinaryNode{PartitionDataType{T}}}(undef, 0)

    i = 0
    for node in AbstractTrees.Leaves(root)
        
        # label.
        i += 1
        node.data.index = i
        
        # get X_parts.
        push!(X_parts, X[node.data.global_X_indices])
        push!(X_parts_inds, node.data.global_X_indices)
        
        # sanity check.
        if !isempty(node.data.X)
            @assert norm(node.data.X-X_parts[end]) < 1e-10
        end
        
        #push!(node, leaves) # cache for speed.
    end

    return X_parts, X_parts_inds
end

# might need to include other data, like kernel matrix, etc. at the leaf nodes.
"""
X_p is X associated with parent.
If the input level value is 1, then kid is a leaf node.
"""
function createchildren(parent,
    left_indicators, direction, X_p::Vector{Vector{T}}, X_p_inds::Vector{Int},
    level::Int;
    store_X_flag::Bool = false) where T

    ## prepare children data.
    X_kid = Vector{Vector{T}}(undef, 0)
    X_kid_inds = Vector{Int}(undef, 0)

    if direction == "left"

        X_kid = X_p[left_indicators]
        X_kid_inds = X_p_inds[left_indicators]
        data = PartitionDataType(HyperplaneType{T}(), X_kid, X_kid_inds, 0)

        kid = leftchild!(parent, data)

    else
        right_indicators = .! left_indicators
        X_kid = X_p[right_indicators]
        X_kid_inds = X_p_inds[right_indicators]
        data = PartitionDataType(HyperplaneType{T}(), X_kid, X_kid_inds, 0)

        kid = rightchild!(parent, data)
    end

    if level == 1
        ## kid is a leaf node. Stop propagation.

        if !store_X_flag
            # do not store inputs leaf nodes.
            kid.data.X = Vector{Vector{T}}(undef, 0)
        end
        
        return nothing
    end

    ## kid is not a leaf node. Propagate.

    # do not store inputs and input info at non-leaf nodes. It is not used during query: i.e., findpartition().
    kid.data.X = Vector{Vector{T}}(undef, 0)
    kid.data.global_X_indices = Vector{Int}(undef, 0)

    # get hyperplane.
    hp_kid, left_indicators_kid = gethyperplane(X_kid)
    kid.data.hp = hp_kid

    createchildren(kid, left_indicators_kid, "left", X_kid, X_kid_inds, level-1)
    createchildren(kid, left_indicators_kid, "right", X_kid, X_kid_inds, level-1)

    return nothing
end

# #### get all leaves. no longer needed because there is a AbstractTrees.leaves() iterator.
# function buildXpart!(X_parts::Vector{Vector{Vector{T}}}, p::BinaryNode{PartitionDataType{T}}) where T

#     #
#     if !isdefined(p, :left) && !isdefined(p, :right)
#         # p is a leaf node. Add its X to X_parts.
#         push!(X_parts, p.data.X)

#         return nothing
#     end

#     # call itself to traverse again.
#     if isdefined(p, :left)
#         buildXpart!(X_parts, p.left)
#     end

#     if isdefined(p, :right)
#         buildXpart!(X_parts, p.right)
#     end
    
#     return nothing
# end

###### search-related.

# Time critical.
"""
Given a point and the tree, find the leaf node of the tree that corresponds to the region that contains this point.
"""
function findpartition(x::Vector{T},
    root::BinaryNode{PartitionDataType{T}},
    levels::Int) where T

    node = root
    for l = 1:levels-1
        if dot(node.data.hp.v, x) < node.data.hp.c
            node = node.left
        else
            node = node.right
        end
    end

    return node.data.index
end


"""
Given a point the tree setup, find the region it is in, and all regions it is within ε-radius to their boundaries.
Recursive function.
"""
function findεpartitions!(region_list::Vector{Int},
    x::Vector{T},
    node::BinaryNode{PartitionDataType{T}},
    levels::Int,
    ε::T) where T

    ### termination case.
    if !isdefined(node, :left) && !isdefined(node, :right)
        # node is the leaf node of the current region.
        push!(region_list, node.data.index)

        return nothing
    end

    ### call itself to traverse again.
    
    hyperplane_eval = dot(node.data.hp.v, x)

    if hyperplane_eval < node.data.hp.c + ε

        findεpartitions!(region_list, x, node.left, levels, ε)
    end

    if hyperplane_eval > node.data.hp.c - ε

        findεpartitions!(region_list, x, node.right, levels, ε)
    end

    return nothing
end


function organizetrainingsets(root::BinaryNode{PartitionDataType{T}},
    levels::Int,
    X0::Vector{Vector{T}},
    ε::T) where T
    
    N = length(X0)
    
    # debug.
    problematic_inds = Vector{Vector{Int}}(undef, 0) # for debug.
    regions_list_set = Vector{Vector{Int}}(undef, N)

    # set up training set.
    N_regions = sum( 1 for node in AbstractTrees.Leaves(root) )
    X_set = Vector{Vector{Vector{T}}}(undef, N_regions)
    X_set_inds = Vector{Vector{Int}}(undef, N_regions)

    for r = 1:N_regions
        X_set[r] = Vector{Vector{T}}(undef, 0)
        X_set_inds[r] = Vector{Int}(undef, 0)
    end

    # build training set by inspecting every input x ∈ X0.
    for n = 1:N

        x = X0[n]
        
        # sanity-check: list of regions must have the region x is in.
        #x_region_ind = findpartition(x, root, levels)
        
        regions_list = Vector{Int}(undef, 0)
        findεpartitions!(regions_list, x, root, levels, ε)

        # add to training set.
        for m = 1:length(regions_list)
            r::Int = regions_list[m]

            if 0 < r <= length(X_set)
                push!(X_set[r], x)
                push!(X_set_inds[r], n)
            else
                # this shouldn't happen.
                # store for debug.
                push!(problematic_inds, n)
            end
        end


        # if length(regions_list) > 1
        #     @assert 1==234
        # end

        # store for debug.
        regions_list_set[n] = regions_list
    end

    return X_set, X_set_inds, regions_list_set, problematic_inds
end