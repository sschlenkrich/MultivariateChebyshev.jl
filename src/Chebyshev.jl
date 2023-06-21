

"""
    chebyshev_transform(
        x::AbstractVecOrMat,
        a::Union{AbstractVector, Number} = -1.0,
        b::Union{AbstractVector, Number} = 1.0,
        )

Transform input x from standard domain `[-1, 1]` to general
hyper-rectangular domain.
"""
function chebyshev_transform(
    x::AbstractVecOrMat,
    a::Union{AbstractVector, Number} = -1.0,
    b::Union{AbstractVector, Number} = 1.0,
    )
    #
    if isa(a, AbstractVector)
        a = a'  # ensure proper broadcasting for x
    end
    if isa(b, AbstractVector)
        b = b'
    end
    return a .+ 0.5 .* (b .- a) .* (x .+ 1.0)
end

"""
    chebyshev_inverse_transform(
        y::AbstractVecOrMat,
        a::Union{AbstractVector, Number} = -1.0,
        b::Union{AbstractVector, Number} = 1.0,
        )

Transform input y from general hyper-rectangular domain to
standard domain `[-1, 1]`.
"""
function chebyshev_inverse_transform(
    y::AbstractVecOrMat,
    a::Union{AbstractVector, Number} = -1.0,
    b::Union{AbstractVector, Number} = 1.0,
    )
    #
    if isa(a, AbstractVector)
        a = a'  # ensure proper broadcasting for x
    end
    if isa(b, AbstractVector)
        b = b'
    end
    return 2 ./ (b.-a) .* (y.-a) .- 1.0
end


"""
    chebyshev_points(degrees::AbstractVector{<:Integer})

Calculate the Chebyshev points of second kind used for interpolation.

`degree` is a list if integers. Each entry represents the maximum polynomial
degree per dimension, length(degrees) corresponds to the number of dimensions

Note that Chebyshev points of second represent the extrema of Chebyshev
polynomials of first kind.
"""
function chebyshev_points(degrees::AbstractVector{<:Integer})
    return [ cos.(pi .* collect(range(0,stop=1,length=n+1))) for n in degrees ]
end


"""
Calculate multivariate Chebyshev points on a standard domain [-1,1].

`degree` is a list if integers. Each entry represents the maximum polynomial
degree per dimension, length(degrees) corresponds to the number of dimensions

Returns a matrix of size (N, D).

Number of rows N equals product over all (N_d + 1) where N_d is
the maximum polynomial degree in dimension d (d=1...D) and D is
the number of dimensions of the tensor, i.e. number of axes of `Array`.
"""
function chebyshev_multi_points(degrees::AbstractVector{<:Integer})
    return cartesian_product(chebyshev_points(degrees))
end


"""
Calculate the Chebyshev polynomials T_0(x), ..., T_N(x)
for up to maximum degree N equal to `max_degree`.

Input `x` can be a float or a vector in the range `[-1, 1]` (element-wise).

Returns a float if input `x` is a float or a vector if input `x` is a
vector.
"""
function chebyshev_polynomials(
    x::Union{AbstractVector, Number},
    max_degree::Integer,
    )
    #
    T = [ 1.0 .+ 0 .* x ]
    if max_degree==0
        return hcat(T...)
    end
    push!(T, x)
    if max_degree==1
        return hcat(T...)
    end
    for d in 2:max_degree
        push!(T, 2*x.*T[end]-T[end-1])
    end
    return hcat(T...)
end


"""
    chebyshev_batch_call(
       C::AbstractArray,
       x::AbstractMatrix,
       )

Calculate

z = [...[C * T(x_D)] * ...] * T(x_1)

for a tensor `C` and input points `x`.

T(x_d) are Chebyshev polynomials to degree N_d.

Input `C` is an n-dim `Array` with suitable size. `size(C) .- 1` specifies
the maximum polynomial degrees per dimension.

Input `x` is a matrix of size (N, D). First dimension N represents batch
size and second dimension D represents number of dimensions of tensor.

Returns a vector of size (N,).

This is the basic operation for calibration and and interpolation.

Re-shaping is to ensure proper broadcast in multiplication.
"""
function chebyshev_batch_call(
    C::AbstractArray,
    x::AbstractMatrix,
    )
    #
    degrees = [ d-1 for d in size(C) ]
    @assert length(size(x))==2
    @assert size(x, 2)==length(degrees)
    res = reshape(C, (size(C)..., 1))
    for d in 1:size(x, 2)
        T = chebyshev_polynomials(x[:,d], degrees[d])
        shape = append!([1], [size(T,2)], [ 1 for k in 1:length(size(res))-3 ], [size(T,1)])
        T = reshape(transpose(T), Tuple(shape))
        if length(size(res))==2  # last iteration
            res = reshape(res, (size(res, 1), 1, size(res, 2)))
        end
        res = batchmul(T, res)
        res = reshape(res, size(res)[2:end])
    end
    return reshape(res, (size(res)[end],))
end


"""
    chebyshev_batch_call(
       C::AbstractArray,
       x::AbstractMatrix,
       matmul::Function,
       )

Calculate

z = [...[C * T(x_D)] * ...] * T(x_1)

for a tensor `C` and input points `x`.

T(x_d) are Chebyshev polynomials to degree N_d.

Input `C` is an n-dim `Array` with suitable size. `size(C) .- 1` specifies
the maximum polynomial degrees per dimension.

Input `x` is a matrix of size (N, D). First dimension N represents batch
size and second dimension D represents number of dimensions of tensor.

Input `matmul` is a Python-like matmul function with multiplication along
the last two dimension and broadcasting along the remaining first dimensions.

Returns a vector of size (N,).
"""
function chebyshev_batch_call(
    C::AbstractArray,
    x::AbstractMatrix,
    matmul::Function,
    )
    #
    degrees = [ d-1 for d in size(C) ]
    @assert length(size(x))==2
    @assert size(x, 2)==length(degrees)
    res = reshape(C, (1, size(C)...))
    for d in size(x, 2):-1:1
        T = chebyshev_polynomials(x[:,d], degrees[d])
        shape = append!([size(T,1)], [ 1 for k in 1:length(size(res))-3 ], [size(T,2)], [1])
        T = reshape(T, Tuple(shape))
        if length(size(res))==2  # last iteration
            res = reshape(res, (size(res, 1), 1, size(res, 2)))
        end
        res = matmul(res, T)
        res = reshape(res, size(res)[1:end-1])
    end
    return reshape(res, (size(res)[1],))
end


"""
    chebyshev_coefficients(
        degrees::AbstractVector{<:Integer},
        multi_points::AbstractMatrix,
        values::AbstractVector,
        matmul::Union{Function, Nothing} = nothing,
        )

Calculate coefficients of Chebyshev basis functions.

`degree` is a list if integers. Each entry represents the maximum polynomial
degree per dimension, length(degrees) corresponds to the number of dimensions.

`multi_points` is a matrix of multivariate Chebyshev points on a standard
domain [-1,1].

`values` is a vector representing the target function values for each
D-dimensional multivariate Chebyshev point on the transformed domain of the
tagret function.

`matmul` may be a Python-like matmul function. Alternatively, `batchmul`
is used for tensor multiplication.

Method returns an `Array` of size `degrees .+ 1`.
"""
function chebyshev_coefficients(
    degrees::AbstractVector{<:Integer},
    multi_points::AbstractMatrix,
    values::AbstractVector,
    matmul::Union{Function, Nothing} = nothing,
    )
    #
    @assert length(size(multi_points)) == 2
    @assert length(size(values)) == 1
    @assert size(multi_points, 1) == prod([n+1 for n in degrees])
    @assert size(multi_points, 1) == size(values, 1)
    @assert size(multi_points, 2) == length(degrees)
    #
    idx_list = [ 1:n+1 for n in degrees ]
    multi_idx = cartesian_product(idx_list)
    inner_node_factor = sum((1 .< multi_idx) .&& (multi_idx .< (degrees' .+ 1)), dims=2)
    values_adj = values .* 0.5.^(length(degrees) .- inner_node_factor)
    values_adj = reshape(values_adj, Tuple([n+1 for n in degrees])) # as tensor
    if isnothing(matmul)
        multi_coeff = chebyshev_batch_call(values_adj, multi_points)
    else
        multi_coeff = chebyshev_batch_call(values_adj, multi_points, matmul)
    end
    multi_coeff = multi_coeff .* (2 .^ inner_node_factor) ./ prod(degrees)
    coeff = reshape(multi_coeff, Tuple([n+1 for n in degrees])) # as tensor
    return coeff
end


"""
Calcuate multivariate Chebyshev interpolation.

Input `y` is a matrix of size (N, D) where N is batch size and D is the number
of dimensions. Input is assumed from general hyper-rectangular domain.

`coeff` is an n-dim array representing the calibrated Chebyshev tensor coefficients.

`a` and `b` are float or vector representing the lower and upper boundaries of
the interpolation domain.
If `a` or `b` is an array then we require `length(a|b)` and equal to `size(y)[end]`.

`matmul` may be a Python-like matmul function. Alternatively, `batchmul`
is used for tensor multiplication.

Method returns a vector of size (N,).
"""
function chebyshev_interpolation(
    y::AbstractMatrix,
    coeff::AbstractArray,
    a::Union{AbstractVector, Number} = -1.0,
    b::Union{AbstractVector, Number} = 1.0,
    matmul::Union{Function, Nothing} = nothing)
    x = chebyshev_inverse_transform(y, a, b)
    if matmul == nothing
        res = chebyshev_batch_call(coeff, x)
    else
        res = chebyshev_batch_call(coeff, x, matmul)
    end
    return res
end


"""
`Chebyshev` represents an object that allows calculation of the multi-variate
Chebyshev polynomial.

This object is typically the result of a calibration.
"""
struct Chebyshev
    coeff::AbstractArray
    a::Union{AbstractVector, Number}
    b::Union{AbstractVector, Number}
    matmul::Union{Function, Nothing}
end

