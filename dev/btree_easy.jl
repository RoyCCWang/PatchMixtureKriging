using AbstractTrees


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

function leftchild(data, parent::BinaryNode)
    !isdefined(parent, :left) || error("left child is already assigned")
    node = typeof(parent)(data, parent)
    parent.left = node
end
function rightchild(data, parent::BinaryNode)
    !isdefined(parent, :right) || error("right child is already assigned")
    node = typeof(parent)(data, parent)
    parent.right = node
end







## Things we need to define
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

## Things that make printing prettier
AbstractTrees.printnode(io::IO, node::BinaryNode) = print(io, node.data)

## Optional enhancements
# These next two definitions allow inference of the item type in iteration.
# (They are not sufficient to solve all internal inference issues, however.)
Base.eltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = BinaryNode{T}
Base.IteratorEltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = Base.HasEltype()

## Let's test it. First build a tree.
root = BinaryNode(0)
l = leftchild(1, root) # technically, this mutates root.
r = rightchild(2, root)
lr = rightchild(3, l)

print_tree(root)
collect(PostOrderDFS(root))
@static if isdefined(@__MODULE__, :Test)
    @testset "binarytree_easy.jl" begin
        @test [node.data for node in PostOrderDFS(root)] == [3, 1, 2, 0]
        @test [node.data for node in PreOrderDFS(root)] == [0, 1, 3, 2]
        @test [node.data for node in Leaves(root)] == [3, 2]
    end
end



