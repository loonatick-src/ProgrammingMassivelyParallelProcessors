module SpMVSeqTest

using ProgrammingMassivelyParallelProcessors

using Test

A = [1 0 0 2
     0 3 0 0
     4 0 5 0]

# should compile and run
A_csr = SparseMatrixCSR(A)

A_dense = Matrix(A_csr)

@test A_dense == A

@test A_csr.nzval == collect(1:5)
@test A_csr.colval == [1, 4, 2, 1, 3]
@test A_csr.rowptr == [1,3,4,6]

# an empty row
A = [1 0 0 2
     0 0 0 0
     3 0 4 0]

A_csr = SparseMatrixCSR(A)

@test A_csr.nzval == collect(1:4)
@test A_csr.colval == [1, 4, 1, 3]
@test A_csr.rowptr == [1, 3, 3, 5]


using LinearAlgebra: mul!

x = ones(eltype(A), size(A, 2))
b = similar(x, size(A, 1))

mul!(b, A_csr, x)

@test b == A * x

end # module
