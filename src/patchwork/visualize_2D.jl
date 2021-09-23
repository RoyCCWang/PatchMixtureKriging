#### visualize 2D.

# dot(u,x)+c = 0 to y = m*x + b.
function get2Dline(u::Vector{T}, c::T) where T

    m = -u[1]/u[2]
    b = c/u[2]

    return m, b
end

# traverse from the root towards the leaves.
# as we traverse, build up the boundary visualizations (t, y).
function getpartitionlines!(y_set::Vector{Vector{T}}, 
    t_set,
    node::BinaryNode{PartitionDataType{T}},
    level::Int,
    min_t, max_t, max_N_t::Int,
    centroid::Vector{T},
    max_dist::T) where T

    # draw line.
    m, b = get2Dline(node.data.hp.v, node.data.hp.c)
    t = LinRange(min_t, max_t, max_N_t)
    y = m .* t .+ b

    # prune according to distance from centroid.
    coordinates = collect( [t[n]; y[n]] for n = 1:length(t) )
    inds = findall(xx->norm(xx-centroid)<max_dist, coordinates)
    y = y[inds]
    t = t[inds]

    # prune according to constraints imposed by parents to current node.
    y_pruned, t_pruned = prunepartitionline(node, collect(y), collect(t))

    # store.
    push!(y_set, y_pruned)
    push!(t_set, t_pruned)
    # push!(y_set, y)
    # push!(t_set, t)

    # do not recurse at the level before leaf nodes, which is level 1.
    if level != 2

        # recurse.
        getpartitionlines!(y_set, t_set, node.left, level-1, min_t, max_t, max_N_t, centroid, max_dist)
        getpartitionlines!(y_set, t_set, node.right, level-1, min_t, max_t, max_N_t, centroid, max_dist)
    end

    return nothing
end

function prunepartitionline(node, y::Vector{T}, t::Vector{T}) where T

    #
    @assert length(y) == length(t)
    y_pruned = y
    t_pruned = t

    if isdefined(node, :parent)

        # node.
        c = node.parent.data.hp.c
        v = node.parent.data.hp.v

        # depending on whether current node was a left or right child of parent, use different constraints.
        constraint_func = xx->(dot(v,xx) < c)
        if node.parent.right == node

            constraint_func = xx->!(dot(v,xx) < c)
        end

        coordinates = collect( [t[n]; y[n]] for n = 1:length(t) )
        inds = findall(constraint_func, coordinates)
        y_pruned = y[inds]
        t_pruned = t[inds]

        y_pruned, t_pruned = prunepartitionline(node.parent, y_pruned, t_pruned)

    end

    return y_pruned, t_pruned
end



