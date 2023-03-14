module SparseMatrixCSRTest

using ProgrammingMassivelyParallelProcessors

using Test

A = [1 0 0 2
     0 3 0 4
     5 0 0 0]

A_csr = SparseMatrixCSR(A)

@test size(A) == size(A_csr)

for i in 1:size(A,1), j in 1:size(A,2)
    @test A[i,j] == A_csr[i,j]
end

end # module
