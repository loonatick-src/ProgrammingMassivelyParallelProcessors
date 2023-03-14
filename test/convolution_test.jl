using ProgrammingMassivelyParallelProcessors

using CUDA


@testset "CPU reference implementation" begin
    kernel_sizes = 3:2:9
    n = 100
    T = Int
    arr_in = ones(T, 100)
    arr_out = similar(arr_in)
    for m in kernel_sizes
        npad = (m-1)÷2
        mask = ones(T, m)
        convolution_1D!(arr_out, arr_in, mask)
        @test all(arr_out[begin+npad:end-npad] .== m)
    end

    @test_throws DimensionMismatch convolution_1D!(arr_out, arr_in, ones(Int, 4))
    @test_throws DimensionMismatch convolution_1D!(similar(arr_in, n+1), arr_in, ones(Int, 3))
end

@testset "GPU baseline implementation" begin
    n = 1024
    mask_width = 5
    npad = mask_width ÷ 2
    T = Float32

    # allocate and initialize buffers on the device
    dvec_in = CUDA.fill(one(T), n)
    dvec_out = similar(dvec_in)
    dmask = CUDA.fill(one(T), mask_width)

    # kernel launch parameters
    threads_per_block = 1024
    nblocks = cld(n, threads_per_block)
    CUDA.@sync begin
        @cuda threads=threads_per_block blocks=nblocks convolution_1D_basic_kernel!(dvec_out, dvec_in, dmask)
    end
    hvec_out = Vector(dvec_out)
    @test all(hvec_out[begin+npad:end-npad] .== mask_width)
end

@testset "GPU tiling using shared memory" begin
    n = 2048
    mask_width = 15
    npad = mask_width÷2
    T = Float32

    dvec_in = CUDA.fill(one(T), n)
    dvec_out = similar(dvec_in)
    dmask = CUDA.fill(one(T), mask_width)

    threads = n÷2
    tile_size = threads + mask_width - 1
    shmem = tile_size * sizeof(Float32)
    blocks = cld(n, threads)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks shmem=shmem convolution_1D_tiled!(dvec_out, dvec_in, dmask)
    end

    hvec_out = Vector(dvec_out)
    @test all(hvec_out[begin+npad:end-npad] .== mask_width)
end
