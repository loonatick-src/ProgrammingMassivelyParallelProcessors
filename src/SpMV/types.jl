import Base:size, getindex, setindex!, IndexStyle, IndexCartesian

using SparseArrays:AbstractSparseArray
using CUDA
using Adapt
using GPUArraysCore: AbstractGPUArray

abstract type AbstractSparseMatrixCSR{Tv, Ti} <: AbstractSparseArray{Tv, Ti, 2} end

struct SparseMatrixCSR{Tv, Ti <: Integer, ValueContainerType<:AbstractVector, IdxContainerType<:AbstractVector} <: AbstractSparseMatrixCSR{Tv, Ti}
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

Base.size(A::AbstractSparseMatrixCSR) = (A.m, A.n)

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

struct SparseMatrixELLCSR{Tv, Ti, ValueContainerType, IdxContainerType, ColMajor} <: AbstractSparseMatrixCSR{Tv, Ti}
    m::Int
    n::Int
    nzval::ValueContainerType
    colval::IdxContainerType
    ell_width::Ti 

    function SparseMatrixELLCSR(m, n, nzval::AbstractArray, colval::AbstractArray{Ti}, ell_width::I) where {Ti, I<:Integer}
        Tv = eltype(nzval)
        ValueContainerType = typeof(nzval)
        IdxContainerType = typeof(colval)
        new{Tv, Ti, ValueContainerType, IdxContainerType, Val{false}}(m, n, nzval, colval, ell_width)
    end
    function SparseMatrixELLCSR(m, n, nzval::AbstractArray, colval::AbstractArray{Ti}, ell_width::I, col_maj::V) where {Ti, I<:Integer, V<:Val}
        Tv = eltype(nzval)
        ValueContainerType = typeof(nzval)
        IdxContainerType = typeof(colval)
        @assert get_value(col_maj) isa Bool
        new{Tv, Ti, ValueContainerType, IdxContainerType, V}(m, n, nzval, colval, ell_width)
    end

    SparseMatrixELLCSR(A::SparseMatrixCSR) = SparseMatrixELLCSR(A, Val(false))

    function SparseMatrixELLCSR(A::SparseMatrixCSR, ::Val{false})
        @views row_widths = A.rowptr[begin+1:end] .- A.rowptr[begin:end-1]
        ell_width = maximum(row_widths)
        nvals = ell_width * A.m
        m = A.m; n = A.n
        colval = similar(A.colval, nvals)
        nzval = similar(A.nzval, nvals)
        _fill_ell_buffers!(nzval, colval, A, ell_width)
        SparseMatrixELLCSR(m, n, nzval, colval, ell_width)
    end

    function SparseMatrixELLCSR(A::SparseMatrixCSR, ::Val{true})
        @views row_widths = A.rowptr[begin+1:end] .- A.rowptr[begin:end-1]
        ell_width = maximum(row_widths)
        nvals = ell_width * A.m
        m = A.m; n = A.n
        colval_rowmaj = similar(A.colval, nvals)
        nzval_rowmaj = similar(A.nzval, nvals)
        _fill_ell_buffers!(nzval_rowmaj, colval_rowmaj, A, ell_width)
        nzval = transpose(reshape(nzval_rowmaj, ell_width, m))[:]
        colval = transpose(reshape(colval_rowmaj, ell_width, m))[:]
        Tv = eltype(nzval)
        Ti = eltype(colval)
        ValueContainerType = typeof(nzval)
        IdxContainerType = typeof(colval)
        new{Tv, Ti, ValueContainerType, IdxContainerType, Val{true}}(m, n, nzval, colval, ell_width)
    end
end

get_colmaj(::SparseMatrixELLCSR{Tv, Ti, V, C, ColMaj}) where {Tv, Ti, V, C, ColMaj} = ColMaj()

Adapt.@adapt_structure SparseMatrixELLCSR

function cu_sparse_ellcsr(A::SparseMatrixELLCSR)
    nzval = CuArray(A.nzval)
    colval = CuArray(A.colval)
    colmaj = get_colmaj(A) 
    SparseMatrixELLCSR(A.m, A.n, nzval, colval, A.ell_width, colmaj)
end

function _fill_ell_buffers!(nzval, colval, A::SparseMatrixCSR, ell_width)
    Tv = eltype(nzval)
    for row in 1:A.m 
        for row in 1:A.m 
            row_begin = A.rowptr[row]
            row_end = A.rowptr[row+1]-1
            row_nzval_csr = @view A.nzval[row_begin:row_end]
            row_colval_csr = @view A.colval[row_begin:row_end]
            ell_begin = (row-1) * ell_width + 1
            ell_end = ell_begin + ell_width - 1
            row_nzval = @view nzval[ell_begin:ell_end]
            row_colval = @view colval[ell_begin:ell_end]
            nnz = length(row_nzval_csr)
            pad_size = ell_width - nnz
            if length(row_colval_csr) == 0 || first(row_colval_csr) > pad_size
                row_nzval[begin:pad_size] .= zero(Tv)
                row_colval[begin:pad_size] .= 1:pad_size
                row_nzval[pad_size+1:end] .= row_nzval_csr
                row_colval[pad_size+1:end] .= row_colval_csr
            elseif A.n - last(row_colval_csr) >= pad_size
                row_nzval[begin:nnz] .= row_nzval_csr
                row_colval[begin:nnz] .= row_colval_csr
                row_nzval[nnz+1:end] .= zero(Tv)
                row_colval[nnz+1:end] .= (nnz+1):(nnz+pad_size)
            else
                i = firstindex(row_colval_csr)
                for (curr_col, _) in pairs(row_nzval)
                    csr_col = row_colval_csr[i]
                    if csr_col == curr_col || pad_size == 0
                        row_nzval[curr_col] = row_nzval_csr[i]
                        row_colval[curr_col] = csr_col
                        i += 1
                    else pad_size > 0
                        row_nzval[curr_col] = zero(Tv)
                        row_colval[curr_col] = curr_col
                        pad_size -= 1
                    end
                end
                @assert i == length(row_colval_csr) + 1
            end
        end
    end
end

function _fill_ell_buffers!(nzval::AbstractGPUArray, colval::AbstractGPUArray, A, ell_width)
    not_implemented()
end

function Base.getindex(A::SparseMatrixELLCSR{Tv, Ti, ValueContainerType, IdxContainerType, Val{false}}, i0::Integer, i1::Integer) where {Tv, Ti <:Integer, ValueContainerType, IdxContainerType}
    rowptr = (i0 - 1) * A.ell_width + 1
    row_colval = @view A.colval[rowptr:rowptr + A.ell_width-1]
    row_nzval = @view A.nzval[rowptr:rowptr + A.ell_width-1]
    j = searchsortedfirst(row_colval, i1)
    if j > length(row_colval)
        return zero(Tv)
    else 
        return row_nzval[j]
    end
end

function Base.getindex(A::SparseMatrixELLCSR{Tv, Ti, ValueContainerType, IdxContainerType, Val{true}}, i0::Integer, i1::Integer) where {Tv, Ti <:Integer, ValueContainerType, IdxContainerType}
    # PERF: assuming `transpose` and `reshape` are lazy
    nzval = reshape(A.nzval, A.m, A.ell_width)
    colval = reshape(A.colval, A.m, A.ell_width)
    row_nzval = @view nzval[i0,:]
    row_colval = @view colval[i0,:]
    j = searchsortedfirst(row_colval, i1)
    if j > length(row_colval)
        return zero(Tv)
    else
        return row_nzval[j]
    end
end

