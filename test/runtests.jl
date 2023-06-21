using Revise

using Logging
using Test

@info "Start testing MultivariateChebyshev package."

@testset verbose=true "MultivariateChebyshev.jl" begin

    include("matrix_methods.jl")
    include("chebychev.jl")
    include("performance.jl")

end

@info "Finished testing MultivariateChebyshev package."
