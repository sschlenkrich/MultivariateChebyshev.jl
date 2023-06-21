
using MultivariateChebyshev
using Test

@testset "Matrix methods." begin
    
    @testset "Cartesian product." begin
        arrays = [
            rand(2),
            rand(3),
            rand(4),
            rand(5),
            rand(6),
        ]
        C = MultivariateChebyshev.cartesian_product(arrays)
        @test size(C) == (2 * 3 * 4 * 5 * 6, 5)
        arrays = [
            [1, 2],
            [3, 4, 5],
            [6, 7],
        ]
        C = MultivariateChebyshev.cartesian_product(arrays)
        C_ref = [
            1 3 6;
            2 3 6;
            1 4 6;
            2 4 6;
            1 5 6;
            2 5 6;
            1 3 7;
            2 3 7;
            1 4 7;
            2 4 7;
            1 5 7;
            2 5 7;
        ]
        @test C == C_ref
        #
        x = [1, 2]
        y = [1, 2, 3]
        z = [1, 2, 3, 4]
        p = MultivariateChebyshev.cartesian_product([x, y, z]) # call with list and unpack
        # println(size(p))
        # println(p)
        v = 100*p[:,1] + 10*p[:,2] + p[:,3]
        # println(v)
        V = reshape(v, (size(x)[1], size(y)[1], size(z)[1]))
        # println(V)
        V_ref = [
            # [:,:,1]
            111 121 131;
            211 221 231;;;
            # [:,:,2]
            112 122 132;
            212 222 232;;;
            # [:,:,3]
            113 123 133;
            213 223 233;;; 
            # [:,:,4]
            114 124 134;
            214 224 234
        ]
        @test V == V_ref
    end
    


    @testset "matmul multiplications." begin
        A = rand(3,4)
        B = rand(4,5)
        C = MultivariateChebyshev.matmul(A, B)
        @test isapprox(C, A * B, atol=1.0e-15)
        # add more specific tests here...
    end

    @testset "batchmul multiplications." begin
        A = rand(3,4)
        B = rand(4,5)
        C = MultivariateChebyshev.batchmul(A, B)
        @test isapprox(C, A * B, atol=1.0e-15)
        # add more specific tests here...
    end

end
