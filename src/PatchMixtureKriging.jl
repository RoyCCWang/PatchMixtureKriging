module PatchMixtureKriging

    # using FFTW
    #
    # #using Distributed
    # #using SharedArrays
    #
    # #import Random
    # #import Printf
    #
    # import Interpolations
    #
    # import Convex
    # import SCS
    # #import SDPA
    #
    # import RWUtilities
    # import SignalTools
    #
    # import Optim
    # import VisualizationTools
    # import Colors
    # import RWRiemannianOptim
    
    using LinearAlgebra
    import Statistics
    using AbstractTrees

    include("./misc/declarations.jl")
    include("./misc/front_end.jl")

    include("./RKHS/RKHS.jl")
    include("./RKHS/kernel.jl")

    include("./RKHS/patchwork_RKHS.jl")

    include("./RKHS/querying.jl")

    include("./patchwork/partition.jl")
    include("./RKHS/mixtureGP.jl")

    include("./misc/utilities.jl")

    include("./patchwork/visualize_2D.jl")

    ### legacy and future clean-up/expansion.
    #include("./RKHS/ODEkernels.jl")
    ##include("./RKHS/interpolators.jl")
    #include("./RKHS/derivatives/SqExp_derivatives.jl")
    #include("./warp_map/Rieszanalysis.jl")
    #include("./optimization/sdp.jl")
    #include("./misc/utilities.jl")

    export  RKHSProblemType,
            fitRKHS!,
            query!,
            #fitnDdensity,
            #fitpdfviaSDP,
            #fitnDdensityRiemannian,
            constructkernelmatrix,
            evalkernel,
            evalquery,

            # mixGP.
            setuppartition,
            getpartitionlines!,
            organizetrainingsets,
            fetchhyperplanes,
            MixtureGPType,
            MixtureGPDebugType,
            fitmixtureGP!

end # module
