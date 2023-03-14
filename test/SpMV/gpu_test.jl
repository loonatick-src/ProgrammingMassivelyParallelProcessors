module GPUSpMVTest

using ProgrammingMassivelyParallelProcessors

using CUDA
using Test

m = n = 1000;
# very memory-inefficient way of generating sparse matrices
# TODO: consider generating COO format and then converting to CSR 
A = rand(m,n)
x = rand(n)
b = A*x
x_cu = CuArray(x)

A_sparse = map(x -> x = x < 0.1 ? x : zero(typeof(x)), A)
A_csr = SparseMatrixCSR(A_sparse)
A_csr_cu = cu_sparsecsr(A_csr)


@test size(A_csr_cu) == size(A)

b_cu = A_csr_cu * x_cu

@test_broken b == Array(b_cu)

end # module
