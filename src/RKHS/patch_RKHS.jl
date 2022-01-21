#####


function prepcfull(P::Array{Vector{RT},D})::SharedArray{Float64,1} where {RT,D}
    N = sum( prod( length(P[i][d]) for d = 1:length(P[i])) for i = 1:length(P) )

    return SharedArray{Float64}(N)
end

# common kernel.

function fitpatchRKHS!( c_full::SharedArray{T,1},
                        X_full_nD::Array{Vector{T},D},
                        y_full::SharedArray{T,D},
                        P,
                        θ::KT,
                        σ²) where {T,D,KT}

    #
    workerfunc! = ii->fitpatchRKHSworker!(c_full,
                            X_full_nD,
                            y_full,
                            P,
                            θ,
                            σ²,
                            ii)

    pmap(workerfunc!, collect(1:length(P)))

    return nothing
end

function fitpatchRKHSworker!( c_full::SharedArray{T,1},
                        X_full_nD::Array{Vector{T},D},
                        y_full::SharedArray{T,D},
                        P,
                        θ::KT,
                        σ²,
                        𝑖::Int) where {T,D, KT}

    # quantities that won't change.
    X = vec(X_full_nD[P[𝑖]...])
    #x_ranges = P[𝑖]
    #X_nD = Utilities.ranges2collection(x_ranges, Val(D))
    #X = vec(X_nD)

    N = length(X)
    w_X = Vector{T}(undef, N)

    # kernel matrix.
    K = Matrix{T}(undef, N, N)
    constructkernelmatrix!(K,
                                w_X,
                                X,
                                θ)

    #
    for i = 1:N
        K[i,i] += σ²[1]
    end

    # compute st, fin.
    st = 1
    if 𝑖 > 1
        st = 1 + sum( prod( length(P[i][d]) for d = 1:length(P[i])) for i = 1:𝑖-1 )
    end
    fin = st + N -1

    c_full[st:fin] = K\vec(y_full[P[𝑖]...])

    return nothing
end



function querypatchRKHS!(   fq::SharedArray{T,D},
                            c_full::SharedArray{T,1},
                            X_full_nD::Array{Vector{T},D},
                            P,
                            θ::KT,
                            Xq_full_nD::Array{Vector{T},D},
                            J) where {T,D,KT}

    #
    workerfunc! = ii->querypatchRKHSworker!(fq,
                            c_full,
                            X_full_nD,
                            P,
                            θ,
                            Xq_full_nD,
                            J,
                            ii)

    pmap(workerfunc!, collect(1:length(P)))

    return nothing
end

"""
P contain the ranges for the observed image.
J contain the ranges for the query image.
"""
function querypatchRKHSworker!( fq::SharedArray{T,D},
                        c_full::SharedArray{T,1},
                        X_full_nD::Array{Vector{T},D},
                        P,
                        θ::KT,
                        Xq_full_nD::Array{Vector{T},D},
                        J,
                        𝑖::Int) where {T,D, KT}

    # quantities that won't change.
    X = vec(X_full_nD[P[𝑖]...])
    Xq_nD = Xq_full_nD[J[𝑖]...]

    N = length(X)
    w_X::Vector{T} = θ.warpfunc[1].(X)

    # compute st, fin.
    st = 1
    if 𝑖 > 1
        st = 1 + sum( prod( length(P[i][d]) for d = 1:length(P[i])) for i = 1:𝑖-1 )
    end
    fin = st + N -1
    c = c_full[st:fin]

    fq_a = xx->evalquery(xx, c, X, w_X, θ)

    fq[J[𝑖]...] = fq_a.(Xq_nD)

    return nothing
end


# function getXcoords(P, X::Vector{Vector{T}}) where T
#     #
#
#     # interior patches.
#     X_set = Vector{Vector{LinRange{T,Int}}}(undef, size(P))
#     for j = 1:size(P,2)
#         for i = 1:size(P,1)
#             out[i,j] = X[P[i,j]...]
#
#     return
# end

function ind2sub(a, i)
    i2s = CartesianIndices(a)
    return i2s[i]
end

function extractnonoverlapboundaries(  P::Array{Vector{UnitRange{Int}},D},
                                        overlap::Int,
                                        y_sz,
                                        dummy_val::T) where {T,D}
    #
    U = Array{Vector{Vector{T}},D}(undef, size(P))


    for n = 1:length(U)

        U[n] = Vector{Vector{T}}(undef,D)
        for d = 1:D
            st = P[n][d][1]
            fin = P[n][d][end]

            u_st = st + overlap/2 - 0.5
            if st == 1
                u_st = 1
                #u_st = -Inf
            end

            u_fin = fin - overlap/2 + 0.5
            if fin == y_sz[d]
                u_fin = y_sz[d]
                #u_fin = Inf
            end

            U[n][d] = [u_st; u_fin]
        end
    end

    return U
end

# ϵ is numerical tolerance.
function isin(xq::Vector{T}, patch_ranges, x_ranges, y_sz; ϵ::T = eps(T)*2) where T
    D = length(xq)
    @assert D == length(patch_ranges) == length(x_ranges)

    for d = 1:D

        st = round(Int,max(patch_ranges[d][1],1))
        fin = round(Int,min(patch_ranges[d][end], y_sz[d]))

        xd_st = x_ranges[d][st]
        xd_fin = x_ranges[d][fin]
        if !(xd_st - ϵ < xq[d] < xd_fin + ϵ)
            return false
        end
    end

    return true
end

# to do: returns closest patch if extrapolation case.
# for now: interpolation only.
function findpatch(xq::Vector{T}, U, y_sz, x_ranges)::Int where T
    N_patches = length(P)
#println("hi")
    for i = 1:N_patches

        if isin(xq, U[i], x_ranges, y_sz)
            return i
        end
    end

    println("Warning: Extrapolation case detected. Returning last patch.")
    return N_patches
end

function prepXqset( Xq,
                    U::Array{Vector{Vector{T}},D},
                    y_sz,
                    x_ranges) where {T,D}

    Nq = length(Xq)

    Xq_set = Array{Vector{Vector{T}},D}(undef, size(U))
    𝓘q_set = Array{Vector{Int},D}(undef, size(U))
    for n = 1:length(Xq_set)
        Xq_set[n] = Vector{Vector{T}}(undef, 0)
        𝓘q_set[n] = Vector{Int}(undef,0)
    end


    for i = 1:Nq
        #n = 0
        #println("i = ", i, ", n = ", n)
        n = findpatch(Xq[i], U, y_sz, x_ranges)

        push!(Xq_set[n], Xq[i])
        push!(𝓘q_set[n], i)
    end

    return Xq_set, 𝓘q_set
end

function parseyq(   sol,
                    𝓘q_set::Array{Vector{Int},D},
                    Nq_array) where {T,D}

    yq = Array{Float64,D}(undef, Nq_array...)

    for i = 1:length(sol)
        𝑖 = sol[i][2]
        patch_query = sol[i][1]

        for j = 1:length(patch_query)
            yq[𝓘q_set[𝑖][j]] = patch_query[j]
        end
    end

    return yq
end

# function reassemblequery(   Xq_set::Array{Vector{Vector{T}},D},
#                             𝓘q_set::Array{Vector{Int},D},
#                             yq_vec) where {T,D}
#     #
#     for n = 1:length(𝓘q_set)
#         patch = 𝓘q_set[n]
#
#         for i = 1:length(patch)
#             k = 𝓘q_set[n][i]
#             yq[k] = Xq
#         end
#     end
#
#     return 0
# end

# speed up idea: make X_full_nD into a Matrix.
function querypatchRKHS2(   Nq_array,
                            c_full::SharedArray{T,1},
                            #X_full_nD::Array{Vector{T},D},
                            X_set,
                            P::Array{Vector{UnitRange{Int}},D},
                            Xq_set,
                            U,
                            θ::KT,
                            𝓘q_set) where {T,D,KT}

    #
    N_patches = length(P)
    @assert size(U) == size(P) == size(Xq_set) == size(𝓘q_set)

    # c_set = Vector{Vector{T}}(undef, N_patches)
    # for 𝑖 = 1:N_patches
    #     st = 1
    #     if 𝑖 > 1
    #         st = 1 + sum( prod( length(P[i][d]) for d = 1:length(P[i])) for i = 1:𝑖-1 )
    #     end
    #     N = prod( length(P[𝑖][d]) for d = 1:D )
    #     fin = st + N -1
    #
    #     c_set[𝑖] = c_full[st:fin]
    # end

    # workerfunc! = ii->querypatchRKHSworker2(
    #                         c_full,
    #                         #c_set,
    #                         #X_full_nD,
    #                         X_set,
    #                         P,
    #                         Xq_set,
    #                         U,
    #                         θ,
    #                         𝓘q_set,
    #                         ii)
    #
    # sol = pmap(workerfunc!, collect(1:length(P)))
    # yqx = parseyq(sol, 𝓘q_set, Nq_array)
    #
    # return yqx



    yq = SharedArray{Float64}(Nq_array...)
    fill!(yq,Inf)
    workerfunc! = ii->querypatchRKHSworker3!( yq,
                            c_full,
                            X_set,
                            P,
                            Xq_set,
                            U,
                            θ,
                            𝓘q_set,
                            ii)
    pmap(workerfunc!, collect(1:length(P)))

    yqx = convert(Array{T,D}, yq)
    return yqx
end

function querypatchRKHSworker3!( yq::SharedArray{T,D},
                        c_full::SharedArray{T,1},
                        X_set,
                        P,
                        Xq_set::Array{Vector{Vector{T}},D},
                        U,
                        θ::KT,
                        𝓘q_set,
                        𝑖::Int) where {T,D, KT}

    println("Start patch ", 𝑖)

    # quantities that won't change.
    X = X_set[𝑖]
    Xq = Xq_set[𝑖]

    N = length(X)

    w_X::Vector{T} = θ.warpfunc[1].(X)

    # compute st, fin.
    st = 1
    if 𝑖 > 1
        st = 1 + sum( prod( length(P[i][d]) for d = 1:length(P[i])) for i = 1:𝑖-1 )
    end
    fin = st + N -1

    c = c_full[st:fin]
    #c = c_set[𝑖]
    #@assert prod( length(P[𝑖][d]) for d = 1:D ) == N

    # fq_a = xx->evalquery(xx, c, X, w_X, θ)
    #
    # # 𝓘q = 𝓘q_set[𝑖]
    # # for j = 1:length(Xq)
    # #     fq[𝓘q[j]] = fq_a(Xq[j])
    # # end
    #
    # return fq_a.(Xq), 𝑖

    # Nq = length(Xq)
    # out = Vector{T}(undef,Nq)
    # for i = 1:Nq
    #     out[i] = evalquery(Xq[i], c, X, w_X, θ)
    # end
    #
    # return out, 𝑖

    Nq = length(Xq)
    for j = 1:Nq
        yq[𝓘q_set[𝑖][j]] = evalquery(Xq[j], c, X, w_X, θ)
    end
    println("Done patch ", 𝑖)

    return nothing
end

function querypatchRKHSworker2(
                        c_full::SharedArray{T,1},
                        #c_set,
                        #X_full_nD::Array{Vector{T},D},
                        X_set,
                        P,
                        Xq_set::Array{Vector{Vector{T}},D},
                        U,
                        θ::KT,
                        𝓘q_set,
                        𝑖::Int) where {T,D, KT}

    # quantities that won't change.

    #X = vec(X_full_nD[P[𝑖]...])
    X = X_set[𝑖]

    Xq = Xq_set[𝑖]

    N = length(X)


    w_X::Vector{T} = θ.warpfunc[1].(X)

    # compute st, fin.
    st = 1
    if 𝑖 > 1
        st = 1 + sum( prod( length(P[i][d]) for d = 1:length(P[i])) for i = 1:𝑖-1 )
    end
    fin = st + N -1

    c = c_full[st:fin]
    #c = c_set[𝑖]
    #@assert prod( length(P[𝑖][d]) for d = 1:D ) == N

    # fq_a = xx->evalquery(xx, c, X, w_X, θ)
    #
    # # 𝓘q = 𝓘q_set[𝑖]
    # # for j = 1:length(Xq)
    # #     fq[𝓘q[j]] = fq_a(Xq[j])
    # # end
    #
    # return fq_a.(Xq), 𝑖

    Nq = length(Xq)
    out = Vector{T}(undef,Nq)
    for i = 1:Nq
        out[i] = evalquery(Xq[i], c, X, w_X, θ)
    end

    return out, 𝑖
end
