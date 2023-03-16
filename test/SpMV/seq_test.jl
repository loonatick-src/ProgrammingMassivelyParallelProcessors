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

# should compile and run
A_ell = SparseMatrixELLCSR(A_csr)
ell_idxs = A_ell.nzval .!= 0
ell_nzval = A_ell.nzval[ell_idxs]
ell_colval = A_ell.colval[ell_idxs]
@test ell_nzval == A_csr.nzval
@test ell_colval == A_csr.colval

for i in 1:size(A,1), j in size(A,2)
    @test A_ell[i,j] == A_csr[i,j] == A[i,j]
end

# should compile and run
A_ell = SparseMatrixELLCSR(A_csr, Val(true))

end # modulncle
