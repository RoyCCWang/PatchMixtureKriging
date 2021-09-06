# routines for front end.





# fits a positive function via SDP2.
function fitnDdensity(  y::Vector{T},
                        X::Vector{Vector{T}},
                        μ::T, # regularization weight.
                        θ::KT,
                        zero_tol::T,
                        max_iters::Int ) where {T,KT}
    #
    # fit.
    K = constructkernelmatrix(X, θ)
    α_solution = fitpdfviaSDP(y, K, μ, max_iters)
    α = clamp.(α_solution, zero_tol, Inf)
    η = RKHSProblemType(α, X, θ, μ)

    return η, α
end

function fitnDdensityRiemannian(  y::Vector{T},
                                X::Vector{Vector{T}},
                                μ::T, # regularization weight.
                                θ::KT,
                                zero_tol::T,
                                max_iter::Int;
                                α_initial = ones(T, length(y)),     ## initial guess.
                                verbose_flag = false,
                                max_iter_tCG = 100,
                                ρ_lower_acceptance = 0.2, # recommended to be less than 0.25
                                ρ_upper_acceptance = 5.0,
                                minimum_TR_radius::T = 1e-3,
                                maximum_TR_radius::T = 10.0,
                                norm_df_tol = 1e-5,
                                objective_tol = 1e-5,
                                avg_Δf_tol = 0.0, #1e-12 #1e-5
                                avg_Δf_window = 10,
                                max_idle_update_count = 50,
                                g::Function = pp->1.0/(dot(pp,pp)+1.0),
                                𝑟 = 1e-2 ) where {T,KT}
    #
    # fit.
    K = constructkernelmatrix(X, θ)
    α_solution, f_α_array_unused, norm_df_array_unused,
        num_iters_unused = RiemannianOptim.solveRKHSℝpproblem(y, K, μ;
                            max_iter = max_iter,
                            α_initial = α_initial,
                            verbose_flag = verbose_flag,
                            max_iter_tCG = max_iter_tCG,
                            ρ_lower_acceptance = ρ_lower_acceptance,
                            ρ_upper_acceptance = ρ_upper_acceptance,
                            minimum_TR_radius = minimum_TR_radius,
                            maximum_TR_radius = maximum_TR_radius,
                            norm_df_tol = norm_df_tol,
                            objective_tol = objective_tol,
                            avg_Δf_tol = avg_Δf_tol,
                            avg_Δf_window = avg_Δf_window,
                            max_idle_update_count = max_idle_update_count,
                            𝑟 = 𝑟,
                            g = g)

    α = clamp.(α_solution, zero_tol, Inf)
    η = RKHSProblemType(α, X, θ, μ)

    return η, α
end
