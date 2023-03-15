module ProgrammingMassivelyParallelProcessors

include("utilities.jl")

export convolution_1D!, convolution_1D_basic_kernel!, convolution_1D_tiled!
include("convolution.jl")

export SparseMatrixCSR, SparseMatrixELL
include("SpMV.jl")

end
