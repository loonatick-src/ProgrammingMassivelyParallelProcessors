using ProgrammingMassivelyParallelProcessors
using SafeTestsets

@safetestset "Convolution" begin include("convolution_test.jl") end
