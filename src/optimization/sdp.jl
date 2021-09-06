

# equation 6.20 fromn GPML book.
# without using squared.
function fitpdfviaSDP(  y::Vector{T},
                        Kp::Matrix{T},
                        μ::T,
                        max_iterations::Int)::Vector{T} where T <: Real

    # parse.
    N = length(y)

    # set up.
    α = Convex.Variable(N)

    constraints = collect( α[i] >= 0 for i = 1:N )
    obj_expr = ( Convex.sumsquares(Kp*α-y) + μ*Convex.quadform(α, Kp) )
    problem = Convex.minimize(obj_expr, constraints)

    # optimize.
    #@time Convex.solve!(problem, SCS.SCSSolver(max_iters = max_iterations))
    @time Convex.solve!(problem, SCS.Optimizer(max_iters = max_iterations))

    # diagnostics.
    println("status: ", problem.status)


    return vec(α.value)
end
