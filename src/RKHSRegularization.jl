module RKHSRegularization

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
    # import Utilities # https://gitlab.com/RoyCCWang/utilities
    # import SignalTools
    #
    # import Optim

    #import RiemannianOptim # https://gitlab.com/RoyCCWang/riemannianoptim
    using LinearAlgebra
    import Statistics
    #import Colors
    
    using AbstractTrees

    # indirect dependencies.
    #import VisualizationTools

    include("./misc/declarations.jl")
    #include("./misc/utilities.jl")
    include("./misc/front_end.jl")

    include("./RKHS/RKHS.jl")
    include("./RKHS/kernel.jl")

    include("./RKHS/patchwork_RKHS.jl")

    #include("./RKHS/ODEkernels.jl")
    ##include("./RKHS/interpolators.jl")
    include("./RKHS/querying.jl")

    include("./patchwork/partition.jl")
    include("./RKHS/mixtureGP.jl")

    include("./misc/utilities.jl")


    include("./patchwork/visualize_2D.jl")

    #include("./RKHS/derivatives/SqExp_derivatives.jl")

    #include("./warp_map/Rieszanalysis.jl")

    #include("./optimization/sdp.jl")

    export  RKHSProblemType,
            fitRKHS!,
            query!,
            #fitnDdensity,
            fitnDdensityRiemannian,
            constructkernelmatrix,
            evalkernel,
            #fitpdfviaSDP,
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
