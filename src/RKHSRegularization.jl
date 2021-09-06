module RKHSRegularization

    using FFTW

    #using Distributed
    #using SharedArrays
    using LinearAlgebra

    #import Random
    #import Printf

    import Interpolations

    import Convex
    import SCS
    #import SDPA

    import Utilities
    import SignalTools

    import RiemannianOptim
    import Optim

    # indirect dependencies.
    #import VisualizationTools

    include("./misc/declarations.jl")
    include("./misc/utilities.jl")
    include("./misc/front_end.jl")

    include("./RKHS/RKHS.jl")
    include("./RKHS/kernel.jl")
    include("./RKHS/ODEkernels.jl")
    #include("./RKHS/interpolators.jl")
    include("./RKHS/querying.jl")

    include("./RKHS/derivatives/SqExp_derivatives.jl")

    include("./warp_map/Rieszanalysis.jl")

    include("./optimization/sdp.jl")

    export  RKHSProblemType,
            fitRKHS!,
            query!,
            fitnDdensity,
            fitnDdensityRiemannian,
            constructkernelmatrix,
            evalkernel,
            fitpdfviaSDP,
            evalquery

end # module
