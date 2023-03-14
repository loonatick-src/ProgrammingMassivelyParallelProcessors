import LinearAlgebra
import LinearAlgebra:mul!
using LinearAlgebra:dot
using GPUArraysCore: AbstractGPUArray

function LinearAlgebra.mul!(b::V1, A::SpCSR, x::V2) where {V1<:AbstractVector, V2<:AbstractVector, SpCSR<:SparseMatrixCSR}
    for i in 1:A.m
        start = A.rowptr[i]
        stop = A.rowptr[i+1]-1
        row = @view A.nzval[start:stop]
        col_indices = @view A.colval[start:stop]
        rvec = @view x[col_indices]
        b[i] = dot(row, rvec)
    end
    b
end

function LinearAlgebra.mul!(b::VCu1, A::SparseMatrixCSR{Tv, Ti, ValueContainerType, IdxContainerType}, x::VCu2) where {VCu1<:AbstractGPUArray, VCu2<:AbstractGPUArray, Tv, Ti<:Integer, ValueContainerType<:AbstractGPUArray, IdxContainerType<:AbstractGPUArray}
    # TODO: compute kernel launch configuration
    _spmv_csr_kernel(b, A, x)
end

function _spmv_csr_kernel(b, A, x)
end
