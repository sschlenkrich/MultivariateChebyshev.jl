
using BenchmarkTools
using MultivariateChebyshev
using Test

include("test_functions.jl")

@testset "Test performance." begin
    
    @testset "Test matrix multiplication performance." begin
        D = 5  # number of dimensions
        Nd = 5 # size per dimension
        degrees = ( Nd for d in 1:D )
        n_points = prod(degrees)
        A = rand(degrees..., n_points)
        B = permutedims(A, length(size(A)):-1:1)
        b1 = @benchmark MultivariateChebyshev.batchmul($A,$A)
        b2 = @benchmark MultivariateChebyshev.matmul($B,$B)
        display(b1)
        println()
        display(b2)
        println()
    end
    
    @testset "Test Black formula performance." begin
        a = [ 0.5, 0.50, -1.0 ]
        b = [ 2.0, 2.50, +1.0 ]
        # degrees = [ 5, 5, 5 ]
        degrees = [ 10, 10, 10 ]
        #
        multi_points = MultivariateChebyshev.chebyshev_multi_points(degrees)
        Y = MultivariateChebyshev.chebyshev_transform(multi_points, a, b)
        values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
        #
        C1 = MultivariateChebyshev.chebyshev_coefficients(degrees, multi_points, values)
        C2 = MultivariateChebyshev.chebyshev_coefficients(degrees, multi_points, values, MultivariateChebyshev.matmul)
        # println(max(abs.(C1 - C2)...))
        @test max(abs.(C1 - C2)...) < 2.0e-16
        b1 = @benchmark MultivariateChebyshev.chebyshev_coefficients($degrees, $multi_points, $values)
        b2 = @benchmark MultivariateChebyshev.chebyshev_coefficients($degrees, $multi_points, $values, $MultivariateChebyshev.matmul)
        display(b1)
        println()
        display(b2)
        println()
    end

end