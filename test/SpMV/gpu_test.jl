module GPUSpMVTest

using ProgrammingMassivelyParallelProcessors

using CUDA
using Test
using LinearAlgebra: norm
using Base: Fix1

import Base: convert

Base.convert(T::Type) = Fix1(convert, T)

m = n = 32;
# very memory-inefficient way of generating sparse matrices
# TODO: consider generating COO format and then converting to CSR 
# TODO: dangerous as some tests might randomly pass
A = rand(Float32, m,n)
x = rand(Float32, n)

# zero-out approx 90% of the elements
A_sparse = map(x -> x = x < 0.1 ? x : zero(typeof(x)), A)
# GeMV
b_dense = A_sparse * x
A_csr = SparseMatrixCSR(A_sparse)
# SpMV
b_csr = A_csr * x

# TODO: why do they differ? the Flops should be the same as sequential version 
@test_broken b_csr == b_dense
@show norm(b_csr .- b_dense, Inf) / norm(b_dense, Inf)

A_csr_cu = cu_sparsecsr(A_csr)
x_cu = CuArray(x)

@test size(A_csr_cu) == size(A)

b_cu = A_csr_cu * x_cu

b_h = Array(b_cu)

reltol = 1.0e-6
@test norm(b_h .- b_csr, Inf) / norm(b_csr, Inf) < reltol

# convert Float to Int  
A_int = A_sparse .* 1000 .|> round .|> convert(Int32)
x_int = x .* 100 .|> round .|> convert(Int32) 
b_dense = A_int * x_int  
A_csr = SparseMatrixCSR(A_int)
b_csr = A_csr * x_int

# Integer arithmetic must be exact when no overflow
@test b_csr == b_dense

A_sparse_cu = cu_sparsecsr(A_csr)
x_cu = CuArray(x_int)
b_cu = A_sparse_cu * x_cu

@test Array(b_cu) == b_csr
@test Array(x_cu) == x_int

A_ell = SparseMatrixELLCSR(A_csr, Val(false))
# should compile and run
A_ell_cu = cu_sparse_ellcsr(A_ell)

@test Array(A_ell_cu.nzval) == A_ell.nzval
@test Array(A_ell_cu.colval) == A_ell.colval
@test A_ell_cu.ell_width == A_ell.ell_width

b_ell_cu = A_ell_cu * x_cu

@test Array(b_ell_cu) == b_csr

end # module
