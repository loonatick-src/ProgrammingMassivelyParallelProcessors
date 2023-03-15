using SafeTestsets

@safetestset "Sequential SpMV tests" begin include("seq_test.jl") end

@safetestset "GPU SpMV tests" begin include("gpu_test.jl") end
