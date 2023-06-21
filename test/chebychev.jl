
using Distributions
using MultivariateChebyshev
using Test

include("test_functions.jl")

@testset "Chebyshev methods." begin

    @testset "Test chebyshev_points." begin
        degrees = [ 1, 2, 3 ]
        points = MultivariateChebyshev.chebyshev_points(degrees)
        @test size(points)==(3,)
        @test points[1]==[1.0, -1.0]
        @test isapprox(points[2], [1.0, 0.0, -1.0], atol=1.0e-15)
        @test isapprox(points[3], [1.0, 0.5, -0.5, -1.0], atol=1.0e-15)
        # println(points)
    end
    
    @testset "Test chebyshev_multi_points." begin
        degrees = [ 1, 2]
        points = MultivariateChebyshev.chebyshev_multi_points(degrees)
        points_ref = [
          1.0  1.0; 
         -1.0  1.0;
          1.0  0.0; 
         -1.0  0.0;
          1.0 -1.0;
         -1.0 -1.0
        ]
        @test isapprox(points, points_ref, atol=1.0e-16)
    end
    
    @testset "Test chebyshev_polynomials." begin
        polys = MultivariateChebyshev.chebyshev_polynomials(0.5, 3)
        # println(size(polys))
        # println(polys)
        @test size(polys)==(1,4)
        @test polys==[1.0 0.5 -0.5 -1.0]
        #
        x = collect(range(-1.0, 1.0, 5))
        polys = MultivariateChebyshev.chebyshev_polynomials(x, 3)
        # println(size(polys))
        # println(polys)
        polys_ref = [
            1.0 -1.0  1.0 -1.0;
            1.0 -0.5 -0.5  1.0;
            1.0  0.0 -1.0 -0.0;
            1.0  0.5 -0.5 -1.0;
            1.0  1.0  1.0  1.0
        ]
        @test size(polys)==(5,4)
        @test polys==polys_ref
    end
    
    @testset "Test chebyshev_batch_call." begin
        C = ones(2, 3, 4)
        x = ones(5, 3)
        y = MultivariateChebyshev.chebyshev_batch_call(C, x)
        # println(size(y))
        # println(y)
        @test y==24.0*ones(5)
        y = MultivariateChebyshev.chebyshev_batch_call(C, x, MultivariateChebyshev.matmul)
        @test y==24.0*ones(5)
    end
    
    @testset "Test chebyshev_coefficients with batchmul." begin
        degrees = [ 2, 3, 4 ]
        multi_points = MultivariateChebyshev.chebyshev_multi_points(degrees)
        values = ones(size(multi_points, 1))
        coeff = MultivariateChebyshev.chebyshev_coefficients(degrees, multi_points, values)
        coeff_ref = zeros(size(coeff))
        coeff_ref[(1 for d in degrees)...] = 1.0
        # println(coeff)
        @test isapprox(coeff, coeff_ref, atol=5.0e-16)
        #
        a = [ 0.5, 0.01, -1.0 ]
        b = [ 2.0, 0.50, +1.0 ]
        degrees = [ 3, 4, 5 ]
        #
        multi_points = MultivariateChebyshev.chebyshev_multi_points(degrees)
        Y = MultivariateChebyshev.chebyshev_transform(multi_points, a, b)
        values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
        C = MultivariateChebyshev.chebyshev_coefficients(degrees, multi_points, values)
        # println(size(C))
        @test size(C)==(4,5,6)
    end

    @testset "Test chebyshev_coefficients with matmul" begin
        degrees = [ 2, 3, 4 ]
        multi_points = MultivariateChebyshev.chebyshev_multi_points(degrees)
        values = ones(size(multi_points, 1))
        coeff = MultivariateChebyshev.chebyshev_coefficients(degrees, multi_points, values, MultivariateChebyshev.matmul)
        coeff_ref = zeros(size(coeff))
        coeff_ref[(1 for d in degrees)...] = 1.0
        # println(coeff)
        @test isapprox(coeff, coeff_ref, atol=5.0e-16)
        #
        a = [ 0.5, 0.01, -1.0 ]
        b = [ 2.0, 0.50, +1.0 ]
        degrees = [ 3, 4, 5 ]
        #
        multi_points = MultivariateChebyshev.chebyshev_multi_points(degrees)
        Y = MultivariateChebyshev.chebyshev_transform(multi_points, a, b)
        values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
        C = MultivariateChebyshev.chebyshev_coefficients(degrees, multi_points, values, MultivariateChebyshev.matmul)
        # println(size(C))
        @test size(C)==(4,5,6)
    end
    
    @testset "Test Black formula chebyshev_points with batchmul" begin
        a = [ 0.5, 0.01, -1.0 ]
        b = [ 2.0, 0.50, +1.0 ]
        degrees = [ 3, 4, 5 ]
        #
        multi_points = MultivariateChebyshev.chebyshev_multi_points(degrees)
        Y = MultivariateChebyshev.chebyshev_transform(multi_points, a, b)
        values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
        C = MultivariateChebyshev.chebyshev_coefficients(degrees, multi_points, values)
        #
        z = MultivariateChebyshev.chebyshev_interpolation(Y, C, a, b)
        # println(max(abs.(z - values)...))
        @test isapprox(z, values, atol=1.0e-14)
    end

    @testset "Test Black formula chebyshev_points with matmul" begin
        a = [ 0.5, 0.01, -1.0 ]
        b = [ 2.0, 0.50, +1.0 ]
        degrees = [ 3, 4, 5 ]
        #
        multi_points = MultivariateChebyshev.chebyshev_multi_points(degrees)
        Y = MultivariateChebyshev.chebyshev_transform(multi_points, a, b)
        values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
        C = MultivariateChebyshev.chebyshev_coefficients(degrees, multi_points, values, MultivariateChebyshev.matmul)
        #
        z = MultivariateChebyshev.chebyshev_interpolation(Y, C, a, b, MultivariateChebyshev.matmul)
        # println(max(abs.(z - values)...))
        @test isapprox(z, values, atol=1.0e-14)
    end
    
    @testset "Teest Black formula with random points and batchmul." begin
        a = [ 0.5, 0.50, -1.0 ]
        b = [ 2.0, 2.50, +1.0 ]
        degrees = [ 5, 5, 5 ]
        #
        multi_points = MultivariateChebyshev.chebyshev_multi_points(degrees)
        Y = MultivariateChebyshev.chebyshev_transform(multi_points, a, b)
        values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
        C = MultivariateChebyshev.chebyshev_coefficients(degrees, multi_points, values)
        #
        base2 = 13
        y = a' .+ rand(2^base2, 3) .* (b.-a)'
        z = MultivariateChebyshev.chebyshev_interpolation(y, C, a, b)
        z_ref = [ BlackOverK(y[i,:]) for i in 1:size(y,1) ]
        # println(max(abs.(z - z_ref)...))
        @test isapprox(z, z_ref, atol=1.0e-1)
        @test max(abs.(z - z_ref)...) < 7.5e-3
    end
    
    @testset "Teest Black formula with random points and matmul" begin
        a = [ 0.5, 0.50, -1.0 ]
        b = [ 2.0, 2.50, +1.0 ]
        degrees = [ 5, 5, 5 ]
        #
        multi_points = MultivariateChebyshev.chebyshev_multi_points(degrees)
        Y = MultivariateChebyshev.chebyshev_transform(multi_points, a, b)
        values = [ BlackOverK(Y[i,:]) for i in 1:size(Y,1) ]
        C = MultivariateChebyshev.chebyshev_coefficients(degrees, multi_points, values, MultivariateChebyshev.matmul)
        #
        base2 = 13
        y = a' .+ rand(2^base2, 3) .* (b.-a)'
        z = MultivariateChebyshev.chebyshev_interpolation(y, C, a, b, MultivariateChebyshev.matmul)
        z_ref = [ BlackOverK(y[i,:]) for i in 1:size(y,1) ]
        # println(max(abs.(z - z_ref)...))
        @test isapprox(z, z_ref, atol=1.0e-1)
        @test max(abs.(z - z_ref)...) < 7.5e-3
    end
    
    

end