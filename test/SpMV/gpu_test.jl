module GPUSpMVTest

using ProgrammingMassivelyParallelProcessors

using CUDA
using Test
using LinearAlgebra: norm
using Base: Fix1

import Base: convert

Base.convert(T::Type) = Fix1(convert, T)

m = n = 4000;
# very memory-inefficient way of generating sparse matrices
# TODO: consider generating COO format and then converting to CSR 
A = rand(Float32, m,n)
x = rand(Float32, n)

# zero-out approx 90% of the elements
A_sparse = map(x -> x = x < 0.1 ? x : zero(typeof(x)), A)
# SGeMV
b_from_dense = A_sparse * x
A_csr = SparseMatrixCSR(A_sparse)
# SSpMV
b_from_csr = A_csr * x

# TODO: why do they differ?
@test_broken b_from_csr == b_from_dense
@show norm(b_from_csr .- b_from_dense, Inf) / norm(b_from_dense, Inf)

A_csr_cu = cu_sparsecsr(A_csr)
x_cu = CuArray(x)

@test size(A_csr_cu) == size(A)

b_cu = A_csr_cu * x_cu

b_h = Array(b_cu)

reltol = 1.0e-6
@test norm(b_h .- b_from_csr, Inf) / norm(b_from_csr, Inf) < reltol

# convert Float to Int  
A_int = A_sparse .* 1000 .|> round .|> convert(Int32)
x_int = x .* 100 .|> round .|> convert(Int32) 
b_from_dense = A_int * x_int  
A_csr = SparseMatrixCSR(A_int)
b_from_csr = A_csr * x_int

# Integer arithmetic must be exact when no overflow
@test b_from_csr == b_from_dense

A_sparse_cu = cu_sparsecsr(A_csr)
x_cu = CuArray(x_int)
b_cu = A_sparse_cu * x_cu

@test Array(b_cu) == b_from_csr

end # module
