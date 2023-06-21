

"""
    cartesian_product(arrays::AbstractVector{<:AbstractVector})

Calculate the cartesian product of a list of input arrays.

Parameter `arrays` represents a list of input vectors.

Method returns a `Matrix` of size (N, D) where D equals the
number of input vectors and N equals the product of lengths
of input vectors.
"""
function cartesian_product(arrays::AbstractVector{<:AbstractVector})
    return vcat(([e...]' for e in Base.product(arrays...))...)
end


"""
    matmul(A::AbstractArray, B::AbstractArray)

Generalised matrix multiplication along last two dimensions.

This method mimics Numpy's matmul behaviour.
"""
function matmul(A::AbstractArray, B::AbstractArray)
    # first we need to check dimensions
    @assert length(size(A)) >= 2
    @assert length(size(A)) == length(size(B))
    @assert size(A)[end] == size(B)[end-1]
    for d in 1:length(size(A))-2
        @assert size(A, d)==1 || size(B, d)==1 || size(A, d)==size(B, d)
    end
    n_dims = length(size(A))
    C = zeros(max.(size(A)[1:end-2],size(B)[1:end-2])..., size(A, n_dims-1), size(B, n_dims))
    idxs_A = (axes(A,d) for d in 1:n_dims-2)
    idxs_B = (axes(B,d) for d in 1:n_dims-2)
    idxs_C = (axes(C,d) for d in 1:n_dims-2)
    for j in axes(B, n_dims)
        for i in axes(A, n_dims-1)
            for k in axes(A, n_dims)
                @views C[idxs_C...,i,j] += A[idxs_A...,i,k] .* B[idxs_B...,k,j]  # .* ensures proper broadcast
            end
        end
    end
    return C
end


"""
    batchmul(A::AbstractArray, B::AbstractArray)

Generalised matrix multiplication along first two dimensions.
"""
function batchmul(A::AbstractArray, B::AbstractArray)
    # Based on https://stackoverflow.com/questions/57678890/batch-matrix-multiplication-in-julia
    # first we need to check dimensions
    @assert length(size(A)) >= 2
    @assert length(size(A)) == length(size(B))
    @assert size(A)[2] == size(B)[1]
    for d in 3:length(size(A))
        @assert size(A, d)==1 || size(B, d)==1 || size(A, d)==size(B, d)
    end
    C = zeros(size(A, 1), size(B, 2), max.(size(A)[3:end],size(B)[3:end])...)
    for I in CartesianIndices(axes(C)[3:end])
        idx_A = min.(Tuple(I), size(A)[3:end])  # allow broadcasting for A
        idx_B = min.(Tuple(I), size(B)[3:end])  # allow broadcasting for B
        @views mul!(C[:, :, Tuple(I)...], A[:, :, idx_A...], B[:, :, idx_B...])
    end
    return C
end
