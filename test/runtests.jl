using Revise

using Logging
using Test

@info "Start testing MultivariateChebyshev package."

@testset "MultivariateChebyshev.jl" begin

    include("matrix_methods.jl")

end

@info "Finished testing MultivariateChebyshev package."
