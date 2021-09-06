# Methods for using Riesz analysis for building the demonstration warp map.

function higherordermonogenicanalysis(y::Array{T,D},
                                ω_set::Vector{T},
                                pass_band_factor::T,
                                scheme_cmd::String,
                                order::Int) where {T,D}

    N_bands = length(ω_set)

    #### Pre-filter.
    attenuation_factor_at_cut_off = 2 # factor of 2 attenuation at cut-off.
    reciprocal_cut_off_percentage = 1.0 # denominator is 0 to 1.
    σ_c = π/(reciprocal_cut_off_percentage*sqrt(2*log(attenuation_factor_at_cut_off)))

    LP, HP = getGaussianfilters(size(y),σ_c) # 3dB attenuation is approx. 2.355 std dev. for a Gaussian.

    Y = real.(ifft(fft(y).*LP)) # bandlimited version of y.
    residual = real.(ifft(fft(y).*HP))

    #### Split-band analysis.
    getfilterpairfunc = getGaussianfilters

    ϕY, ψY = runsplitbandanalysis(Y, ω_set, getfilterpairfunc)
    ηY = runbandpassanalysis(Y, ω_set, pass_band_factor, getfilterpairfunc)

    #### Riesz transform on the different filtered signals.
    H, ordering = gethigherorderRTfilters(Y,order)

    𝓡ϕY = collect( RieszAnalysisLimited(ϕY[s],H) for s = 1:N_bands)
    𝓡ψY = collect( RieszAnalysisLimited(ψY[s],H) for s = 1:N_bands)
    𝓡ηY = collect( RieszAnalysisLimited(ηY[s],H) for s = 1:N_bands)

    #### Generalized monogenic analysis.
    Hᵤ = Array{Array{Float64,D}}(N_bands)
    Aᵤ = Array{Array{Float64,D}}(N_bands)
    ϕᵤ = Array{Array{Float64,D}}(N_bands)

    # Metric tensor.
    metric_tensor_vec = collect( Combinatorics.multinomial(ordering[m]...) for m = 1:length(ordering) )

    # Perform analysis.
    for ss = 1:N_bands
        Hᵤ[ss] = directionalHilbert(metric_tensor_vec, 𝓡ηY[ss])
        Aᵤ[ss], ϕᵤ[ss] = monogenicanalysis(Hᵤ[ss], ηY[ss])
    end

    #### Compute ϕ.
    ϕ_set = collect( Hᵤ[s].*cos(ϕᵤ[s]) for s = 1:N_bands )

    return ϕ_set, 𝓡ϕY, 𝓡ψY, 𝓡ηY, H, ordering, Aᵤ, ϕᵤ, Hᵤ, Y, y, ϕY, ψY, ηY
end


function getRieszwarpmapsamples( y::Array{T,D},
                            ::Val{:simple},
                            ::Val{:uniform},
                            ω_set,
                            pass_band_factor) where {T,D}
    #
    N_bands = length(ω_set)

    Y = y

    #### Split-band analysis.
    ϕY, ψY = SignalTools.runsplitbandanalysis(Y, ω_set, SignalTools.getGaussianfilters)
    ηY = SignalTools.runbandpassanalysis(Y, ω_set, pass_band_factor, SignalTools.getGaussianfilters)

    # #### Riesz transform on the different filtered signals.
    # H, ordering = gethigherorderRTfilters(Y,order)
    #
    # 𝓡ϕY = collect( RieszAnalysisLimited(ϕY[s],H) for s = 1:N_bands)
    # 𝓡ψY = collect( RieszAnalysisLimited(ψY[s],H) for s = 1:N_bands)
    # 𝓡ηY = collect( RieszAnalysisLimited(ηY[s],H) for s = 1:N_bands)

    ϕ_set = ηY

    ϕ = reduce(+,ϕ_set)./N_bands

    return ϕ
    #return y #ηY[4]
end
