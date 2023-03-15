import Base:size, getindex, setindex!, IndexStyle, IndexCartesian

using SparseArrays:AbstractSparseArray
using CUDA
using Adapt
using GPUArraysCore: AbstractGPUArray

struct SparseMatrixCSR{Tv, Ti <: Integer, ValueContainerType<:AbstractVector, IdxContainerType<:AbstractVector} <: AbstractSparseArray{Tv, Ti, 2}
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
    # WARN: PERF: row-major traversal of column-major matrix
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
    row_colvals = @view A.colval[colidx_begin:colidx_end]
    nzval_idx = searchsortedfirst(row_colvals, j)
    if nzval_idx > length(row_colvals) || row_colvals[nzval_idx] != j
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

struct SparseMatrixELL{Tv, Ti, ValueContainerType, IdxContainerType, ColMajor} <: AbstractSparseArray{Tv, Ti, 2}
    m::Int
    n::Int
    nzval::ValueContainerType
    colval::IdxContainerType
    ell_width::Ti 

    function SparseMatrixELL(m, n, nzval::AbstractArray, colval::AbstractArray{Ti}, ell_width::I) where {Ti, I<:Integer}
        Tv = eltype(nzval)
        ValueContainerType = typeof(nzval)
        IdxContainerType = typeof(colval)
        new{Tv, Ti, ValueContainerType, IdxContainerType, Val(false)}(m, n, nzval, colval, ell_width)
    end

    function SparseMatrixELL(A::SparseMatrixCSR)
        idxs = A.rowptr[1:A.m]
        @views row_widths = idxs[begin+1:end] .- idxs[begin:end-1]
        ell_width = maximum(row_widths)
        nvals = ell_width * A.m
        m = A.m; n = A.n
        colval = similar(A.colval, nvals)
        nzval = similar(A.nzval, nvals)
        _fill_ell_buffers!(nzval, colval, A, ell_width)
        SparseMatrixELL(m, n, nzval, colval, ell_width)
    end
end

Adapt.@adapt_structure SparseMatrixELL

function _fill_ell_buffers!(nzval, colval, A::SparseMatrixCSR, ell_width)
    Tv = eltype(nzval)
    for row in 1:A.m 
        ell_start = (row-1) * ell_width + 1
        start = A.rowptr[row]
        stop = A.rowptr[row+1]-1
        nnz = stop - start + 1
        pad_size = ell_width - (start - stop + 1)
        # TODO: the code should run without this condition as well
        if pad_size == 0
            continue
        end
        row_colval = @view colval[ell_start:ell_start+ell_width-1]
        row_nzval = @view nzval[ell_start:ell_start+ell_width-1]
        @assert length(row_colval) == length(row_nzval) == ell_width
        if A.n - A.colval[stop] >= pad_size # add all the padding at the end
            row_colval[begin:nnz] .= A.colval[start:stop]
            row_nzval[begin:nnz] .= A.nzval[start:stop]
            # remaining is padding
            row_nzval[nnz+1:end] .= zero(Tv)
            padcolval_start = A.colval[stop]+1
            padcolval_stop = padcolval_start + pad_size - 1 
            @show length(row_colval[nnz+1:end]) length(padcolval_start:padcolval_stop)
            row_colval[nnz+1:end] .= padcolval_start:padcolval_stop
        elseif A.colval[start] > pad_size  # add all the padding at the beginning
            row_nzval[begin:pad_size] .= zero(Tv)
            row_colval[begin:pad_size] .= 1:pad_size # TODO: use firstindex, axis etc instead of hardcoding indices
            row_nzval[pad_size+1:end] .= A.nzval[start:stop]
            row_colval[pad_size+1:end] .= A.colval[start:stop]
        else  # interleave using greedy approach
            # TODO: use firstindex and/or axes instead of hardcoding indices 
            current_colval = 1
            k = start
            j = firstindex(row_colval)
            while j <= lastindex(ell_width) && k <= stop
                cval = A.colval[k]
                if current_colval != cval  # insert padding
                    row_nzval[j] = zero(Tv)
                    row_colval[j] = current_colval
                    current_colval += 1
                else  # insert actual value
                    row_nzval[j] = A.nzval[k] 
                    row_colval[j] = cval
                    k += 1
                end
                j += 1
            end
        end
    end
end

function _fill_ell_buffers!(nzval::AbstractGPUArray, colval::AbstractGPUArray, A, ell_width)
    not_implemented()
end
