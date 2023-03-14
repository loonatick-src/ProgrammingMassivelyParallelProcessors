import Base:size, getindex, setindex!, IndexStyle, IndexCartesian

using CUDA
using Adapt

struct SparseMatrixCSR{Tv, Ti <: Integer, ValueContainerType<:AbstractVector, IdxContainerType<:AbstractVector} <: AbstractMatrix{Tv}
    m::Int
    n::Int
    rowptr::IdxContainerType
    colval::IdxContainerType
    nzval::ValueContainerType

    function SparseMatrixCSR(m::I, n::I, rowptr::Vi, colval::Vi, nzval::V) where {I<:Integer, Vi<:AbstractVector{<:Integer}, V<:AbstractVector}
        new{eltype(nzval), Int, typeof(nzval), typeof(rowptr)}(m, n, rowptr, colval, nzval)
    end
end

Adapt.@adapt_structure SparseMatrixCSR

function cu_sparsecsr(A::SparseMatrixCSR)
    m, n = size(A)
    rowptr = CuArray(A.rowptr)
    colval = CuArray(A.colval)
    nzval = CuArray(A.nzval)
    SparseMatrixCSR(m, n, rowptr, colval, nzval)
end

function SparseMatrixCSR(A::Matrix{T}) where {T}
    m, n = size(A)
    nzval = T[]
    rowptr = Int[1]
    colval = Int[]
    for i in 1:m
        nnz = 0
        for j in 1:n 
            Aij = A[i,j]
            if Aij != 0
                nnz += 1
                push!(nzval, Aij)
                push!(colval, j)
            end
        end
        push!(rowptr, rowptr[end]+nnz)
    end
    SparseMatrixCSR(m, n, rowptr, colval, nzval)
end

Base.size(A::SparseMatrixCSR) = (A.m, A.n)

# TODO: consult interface docs for exact interface of AbstractMatrix
function Base.getindex(A::SparseMatrixCSR, i::Int)
    i = ((i-1) ÷ A.n) + 1
    j = ((i-1) % A.n) + 1
    getindex(A, i, j)
end

Base.getindex(A::SparseMatrixCSR, I::Tuple{Int, Int}) = getindex(A, I[1], I[2])
function Base.getindex(A::SparseMatrixCSR, i::Int, j::Int)
    colidx_begin = A.rowptr[i]
    colidx_end = A.rowptr[i+1]-1
    nzval_idx = searchsortedfirst(view(A.colval, colidx_begin, colidx_end), j)
    if nzval_idx > colidx_end - colidx_begin
        return zero(eltype(A))
    else
        return A.nzval[colidx_begin+nzval_idx-1]
    end
end

function Base.setindex!(A::SparseMatrixCSR, i::Int)
    not_implemented()
end

function Base.setindex!(A::SparseMatrixCSR, I::Vararg{Int, ndims(SparseMatrixCSR)})
    not_implemented()
end

function Matrix(A::SparseMatrixCSR)
    A_dense = zeros(eltype(A), size(A))
    for i in 1:A.m
        start = A.rowptr[i]
        stop = A.rowptr[i+1]-1
        js = @view A.colval[start:stop]
        vals = @view A.nzval[start:stop]
        for (j, nzv) in zip(js, vals)
            A_dense[i,j] = nzv
        end
    end
    A_dense
end
