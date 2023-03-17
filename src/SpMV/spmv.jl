import LinearAlgebra
import LinearAlgebra:mul!
import Base:*

using LinearAlgebra:dot
using GPUArraysCore: AbstractGPUVector
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

# FIXME: not being dispatched correctly
#=
@generated function LinearAlgebra.mul!(b::VCu1,
    A::SpCSR, x::VCu2) where {VCu1<:AbstractGPUVector, VCu2<:AbstractGPUVector, SpCSR<:SparseMatrixCSR}
    # TODO: fetch warp size and max threads per block instead of hard-coding values
    threads = min(32*cld(A.m, 32), 1024)
    blocks = cld(A.m, threads)
    CUDA.@sync begin
        if A isa SparseMatrixCSR
            @cuda threads=threads blocks=blocks _spmv_csr_kernel(b, A, x)
        elseif A isa SparseMatrixELLCSR && !get_value(get_colmaj(A))
            @cuda threads=threads blocks=blocks _spmv_ellcsr_kernel(b, A, x)
        else
            @cuda threads=threads blocks=blocks _spmv_ellcsr_kernel_t(b, A, x)
        end
    end
    b
end
=#

function LinearAlgebra.mul!(b::VCu1, A::SparseMatrixCSR{Tv, Ti, VC, IC}, x::VCu2) where {Tv, Ti, VC<:AbstractGPUVector, IC<:AbstractGPUVector{<:Integer}, VCu1<:AbstractGPUVector, VCu2<:AbstractGPUVector}
    threads = min(32*cld(A.m, 32), 1024)
    blocks = cld(A.m, threads)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks _spmv_csr_kernel(b, A, x)
    end
    b
end

function LinearAlgebra.mul!(b::VCu1, A::SparseMatrixELLCSR{Tv, Ti, VC, IC, Val{false}}, x::VCu2) where {Tv, Ti, VC<:AbstractGPUVector, IC<:AbstractGPUVector{<:Integer}, VCu1<:AbstractGPUVector, VCu2<:AbstractGPUVector}
    threads = min(32*cld(A.m, 32), 1024)
    blocks = cld(A.m, threads)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks _spmv_ellcsr_kernel(b, A, x)
    end
    b
end

function LinearAlgebra.mul!(b::VCu1, A::SparseMatrixELLCSR{Tv, Ti, VC, IC, Val{true}}, x::VCu2) where {Tv, Ti, VC<:AbstractGPUVector, IC<:AbstractGPUVector{<:Integer}, VCu1<:AbstractGPUVector, VCu2<:AbstractGPUVector}
    threads = min(32*cld(A.m, 32), 1024)
    blocks = cld(A.m, threads)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks _spmv_ellcsr_kernel_t(b, A, x)
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

function _spmv_ellcsr_kernel(b, A, x)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    acc = zero(eltype(b))
    if i <= A.m
        base = (i-1)*A.ell_width+1
        for k in 1:A.ell_width
            j = A.colval[base+k-1] 
            acc += A.nzval[base+k-1]*x[j]
        end
        b[i] = acc
    end
    nothing
end

function _spmv_ellcsr_kernel_t(b, A, x)
    row = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = A.m
    if row <= A.m
        acc = zero(eltype(x))
        for i in 1:A.ell_width
            colval_idx = (i-1)*stride + row
            j = A.colval[colval_idx]
            acc += A.nzval[colval_idx] * x[j]
        end
        b[row] = acc
    end
    nothing
end

