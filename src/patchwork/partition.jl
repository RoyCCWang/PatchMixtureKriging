
mutable struct HyperplaneType{T}
    v::Vector{T}
    c::T
end

mutable struct PartitionDataType{T}
    hp::HyperplaneType{T}
    X::Vector{Vector{T}}
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
    #
    X_mat = (array2matrix(X))'
    U, s, V = svd(X_mat)
    v = V[:,1]

    indicators, functional_evals, c = splitpoints(v, X)
    hp = HyperplaneType(v, c)

    return hp, indicators
end


"""
current node is p.
"""
function setuppartition(X::Vector{Vector{T}}, level) where T

    # get hyperplane.
    hp, left_indicators = gethyperplane(X)
    X_empty = Vector{Vector{T}}(undef, 0)

    data = PartitionDataType(hp, X_empty)

    # add to current node.
    root = BinaryNode(data)

    # might have to use recursion.
    createchildren(root, hp, left_indicators, "left", X, level-1)
    createchildren(root, hp, left_indicators, "right", X, level-1)
    
    return root
end


# might need to include other data, like kernel matrix, etc. at the leaf nodes.
"""
X_p is X associated with parent.
If the input level value is 1, then kid is a leaf node.
"""
function createchildren(parent,
    hp, left_indicators, direction, X_p::Vector{Vector{T}}, 
    level::Int;
    store_X_for_every_node::Bool = false) where T

    ## prepare children data.
    X_kid = Vector{Vector{T}}(undef, 0)

    if direction == "left"

        X_kid = X_p[left_indicators]
        data = PartitionDataType(hp, X_kid)

        kid = leftchild!(parent, data)

    else
        right_indicators = .! left_indicators
        X_kid = X_p[right_indicators]
        data = PartitionDataType(hp, X_kid)

        kid = rightchild!(parent, data)
    end

    if level == 1
        ## kid is a leaf node. Stop propagation.
        return nothing
    end

    ## kid is not a leaf node. Propagate.

    if !store_X_for_every_node
        # do not store inputs at non-leaf nodes.
        kid.data.X = Vector{Vector{T}}(undef, 0)
    end

    # get hyperplane.
    hp_kid, left_indicators_kid = gethyperplane(X_kid)

    createchildren(kid, hp_kid, left_indicators_kid, "left", X_kid, level-1)
    createchildren(kid, hp_kid, left_indicators_kid, "right", X_kid, level-1)

    return nothing
end

#### get all leaves.
function buildXpart!(X_parts::Vector{Vector{Vector{T}}}, p::BinaryNode{PartitionDataType{T}}) where T

    #
    if !isdefined(p, :left) && !isdefined(p, :right)
        # p is a leaf node. Add its X to X_parts.
        push!(X_parts, p.data.X)

        return nothing
    end

    # call itself to traverse again.
    if isdefined(p, :left)
        buildXpart!(X_parts, p.left)
    end

    if isdefined(p, :right)
        buildXpart!(X_parts, p.right)
    end

    return nothing
end

###### search.

"""
Given a point and the tree, find the leaf node of the tree that corresponds to the region that contains this point.
"""
function myfunc()

    #

end