import LinearAlgebra
import LinearAlgebra:mul!
import Base:*

using LinearAlgebra:dot
using GPUArraysCore: AbstractGPUArray
using CUDA

function Base.:*(A::SpCSR, x::V) where {SpCSR<:AbstractSparseMatrixCSR, V<:AbstractVector}
    b = similar(x)
    mul!(b, A, x)
end

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
    # TODO: fetch warp size and max threads per block instead of hard-coding values
    threads = min(32*cld(A.m, 32), 1024)
    blocks = cld(A.m, threads)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks _spmv_csr_kernel(b, A, x)
    end
    b
end

function _spmv_csr_kernel(b, A, x)
    # fixme
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= A.m
        acc = zero(eltype(b))
        start = A.rowptr[i]
        stop = A.rowptr[i+1]-1
        for k in start:stop
            acc += A.nzval[k] * x[A.colval[k]]
        end
        b[i] = acc
    end
    nothing
end

function LinearAlgebra.mul!(b::VCu_out, A::SparseMatrixELLCSR{Tv, Ti, ValueContainerType, IdxContainerType, Val(false)}, x::VCu_in) where {VCu_out <: AbstractGPUArray, VCu_in<:AbstractGPUArray, Tv, Ti<:Integer, ValueContainerType<:AbstractGPUArray, IdxContainerType<:AbstractGPUArray}
    # TODO: fetch warp size and max threads per block instead of hard-coding values
    threads = min(32*cld(A.m, 32), 1024)
    blocks = cld(A.m, threads)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks _spmv_ellcsr_kernel(b, A, x)
    end
    b
end

function _spmv_ellcsr_kernel(b, A, x)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    acc = zero(eltype(b))
    if i < A.m
        start = (i-1) * (A.ell_width) + 1
        stop = start + A.ell_width - 1
        for k in start:stop
            j = A.colval[k]
            acc += A.nzval[j] * x[j]
        end
        b[i] = acc
    end
    nothing
end
