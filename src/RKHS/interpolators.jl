# Methods that deal with interpolating the adaptive kernel's kernel parameters
#   or the warp map from samples.

# # One single interpolator for the size of the entire image.
# function getθset{KT}(itp, itp_set, θ::KT, scale_factor::Float64,
#                 ratio::Vector{Float64},
#                 offset::Vector{Float64})::Matrix{AdaptiveKernelType{KT}}
#
#     θ_set = Array{AdaptiveKernelType{KT}}(size(itp_set))
#
#     for i = 1:length(θ_set)
#         θ_set[i] = AdaptiveKernelType(θ, x->scale_factor*itp[(x./ratio+offset)...] )
#     end
#
#     return θ_set
# end
#
# # The version that uses an array of smaller interpolators, with oversized phi patches.
# function getθsetlocalinterpolators2{KT}(itp_sets::Vector,
#                 θ::KT, scale_factor::Float64,
#                 ratio::Vector{Float64},
#                 X_set::Matrix{Matrix{Vector{Float64}}},
#                 overlap::Int)::Matrix{AdaptiveKernelType{KT}}
#
#     N_rows, N_cols = size(itp_sets[1])
#     θ_set = Array{AdaptiveKernelType{KT}}(N_rows,N_cols)
#
#     M = length(itp_sets)
#
#     # Top left. Put into for-loop block so anchor and offset gets its own scope.
#     for i = 1:1
#         anchor = [1; 1]
#         offset = -X_set[1,1][1]./ratio + anchor
#
#         warpfunc_set = Vector{Function}(M)
#         for m = 1:M
#             itp_set = itp_sets[m]
#             warpfunc_set[m] = x->scale_factor*itp_set[1,1][(x./ratio+offset)...]
#         end
#
#         θ_set[1,1] = AdaptiveKernelType(θ,warpfunc_set)
#     end
#
#     # Left column, second row until end.
#     for i = 2:N_rows
#         anchor = [overlap+1; 1]
#         offset = -X_set[i,1][1]./ratio + anchor
#
#         warpfunc_set = Vector{Function}(M)
#         for m = 1:M
#             itp_set = itp_sets[m]
#             warpfunc_set[m] = x->scale_factor*itp_set[i,1][(x./ratio+offset)...]
#         end
#
#         θ_set[i,1] = AdaptiveKernelType(θ,warpfunc_set)
#     end
#
#     # Top row, second column until end.
#     for j = 2:N_cols
#         anchor = [1; overlap+1]
#         offset = -X_set[1,j][1]./ratio + anchor
#
#         warpfunc_set = Vector{Function}(M)
#         for m = 1:M
#             itp_set = itp_sets[m]
#             warpfunc_set[m] = x->scale_factor*itp_set[1,j][(x./ratio+offset)...]
#         end
#
#         θ_set[1,j] = AdaptiveKernelType(θ,warpfunc_set)
#     end
#
#     # Central regions, second row ⊗ second column until end.
#     for i = 2:N_rows
#         for j = 2:N_cols
#             anchor = [overlap+1; overlap+1]
#             offset = -X_set[i,j][1]./ratio + anchor
#
#             warpfunc_set = Vector{Function}(M)
#             for m = 1:M
#                 itp_set = itp_sets[m]
#                 warpfunc_set[m] = x->scale_factor*itp_set[i,j][(x./ratio+offset)...]
#             end
#
#             θ_set[i,j] = AdaptiveKernelType(θ,warpfunc_set)
#         end
#     end
#
#     return θ_set
# end



function getwarpmap(ϕ::Array{T,D}) where {T,D}

    itp_ϕ = Interpolations.interpolate(ϕ,
                Interpolations.BSpline(Interpolations.Cubic(
                    Interpolations.Flat(Interpolations.OnGrid()))))

    etp_ϕ = Interpolations.extrapolate(itp_ϕ, Interpolations.Line())
    #etp_ϕ = Interpolations.extrapolate(itp_ϕ, 0)

    return etp_ϕ
end

function getwarpmap(ϕ::Array{T,D}, x_ranges::Vector, amplification_factor::T) where {T,D}

    @assert D == length(x_ranges)
    N_array = collect( length(x_ranges[d]) for d = 1:D )

    itp_ϕ = Interpolations.interpolate(ϕ,
                Interpolations.BSpline(Interpolations.Cubic(
                    Interpolations.Flat(Interpolations.OnGrid()))))

    etp_ϕ = Interpolations.extrapolate(itp_ϕ, Interpolations.Line())
    #etp_ϕ = Interpolations.extrapolate(itp_ϕ, 0)

    st = collect( x_ranges[d][1] for d = 1:D )
    fin = collect( x_ranges[d][end] for d = 1:D )
    f = xx->etp_ϕ(interval2itpindex(xx,
                st,
                fin,
                N_array)...)*amplification_factor

    # chain rule for first derivatives.
    df = xx->( Interpolations.gradient(etp_ϕ,interval2itpindex(xx,
                st,
                fin,
                N_array)...) .* derivativeinterval2itpindex(st,fin,N_array) .*amplification_factor )

    # chain rule for second derivatives.
    d2f = xx->( Interpolations.hessian(etp_ϕ,interval2itpindex(xx,
                st,
                fin,
                N_array)...) .* (derivativeinterval2itpindex(st,fin,N_array).^2) .*amplification_factor )


    return f, df, d2f
end
# # test code.
#
# A = randn(N,N)
# itp_ϕ = Interpolations.interpolate(A,
#             Interpolations.BSpline(Interpolations.Cubic(
#                 Interpolations.Flat(Interpolations.OnGrid()))))
#
# P = [1.1; 2.3]
# ϕ_map_func, d_ϕ_map_func = getwarpmap(A, x_ranges[1:2], amplification_factor)
#
#
# dϕ_ND = xx->Calculus.gradient(ϕ_map_func, xx)
# dϕ_ND(P)
#
# d_ϕ_map_func(P) ./ dϕ_ND(P)
