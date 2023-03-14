using CUDA

using Base.Broadcast: combine_axes

function convolution_1D!(vec_out::Vector{T}, vec_in::Vector{T}, mask::Vector{T}) where T
    length(mask) % 2 != 1 && throw(DimensionMismatch("This 1D convolution kernel does not handle masks with even number size"))
    length(mask) > length(vec_out) && throw(DimensionMismatch("Mask canot be larger than the input and output arrays"))
    combine_axes(vec_in, vec_out)
    _convolution_1D!(vec_out, vec_in, mask)
end

"""
1D Convolution: CPU Implementation

Reference implementation for checking correctness of GPU kernels.

Assumptions:
- output and input have same size
- _implicit_ zero padding of input array
- size of mask is odd

TODO: make a wrapper that checks all these
"""
function _convolution_1D!(vec_out::Vector{T}, vec_in::Vector{T}, mask::Vector{T}) where {T <: Number}
    npad = (length(mask) - 1) ÷ 2
    mid = npad+1
    # left padding
    for i in 1:npad
        vec_out[i] = sum(mask[mid-i+1:end] .* vec_in[begin:npad+i])
    end
    # interior elements
    for (i, _) in pairs(vec_out[begin+npad:end-npad])
        vec_out[i+npad] = sum(mask .* vec_in[i:i+2*npad])
    end
    # right padding
    for i in 1:npad
        idx = lastindex(vec_out) - npad + i
        vec_out[idx] = sum(mask[begin:end-i] .* vec_in[idx-npad:end])
    end
    vec_out
end

"""
Baseline 1D convolution GPU kernel
"""
function convolution_1D_basic_kernel!(dvec_out, dvec_in, mask)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    mask_width = length(mask)
    npad = (mask_width - 1) ÷ 2
    conv_value = zero(eltype(dvec_in))
    base = i - mask_width÷2 - 1
    for j in 1:mask_width
        idx = base + j
        if 0 < idx <= length(dvec_out)
            conv_value += dvec_in[idx] * mask[j]
        end
    end
    dvec_out[i] = conv_value
    return nothing
end

"""1D Convolution using device shared memory
Shared memory has to be the dumbest name NVIDIA came up with IMO.
"""
function convolution_1D_tiled!(dvec_out, dvec_in, mask)
    # FIXME: most likely throwing exception
    # inspect with @device_code_warntype
    mask_width = length(mask)
    T = eltype(dvec_in)
    npad = mask_width >> 1
    tile_size = blockDim().x + length(mask) - 1
    in_tile = CuDynamicSharedArray(T, tile_size)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x     
    dvec_out[i] = 2 * one(T)

    # load left halo/ghost cells into shared memory
    left_pad_idx = i - npad
    if threadIdx().x <= npad
        in_tile[threadIdx().x] = left_pad_idx < 1 ? zero(T) : dvec_in[left_pad_idx]
    end
    # load the elements themselves
    in_tile[threadIdx().x + npad] = dvec_in[i]
    # load padding on the right into shared memory
    right_pad_idx = i + npad
    if threadIdx().x > blockDim().x - npad 
        in_tile[threadIdx().x + 2*npad] = right_pad_idx > length(dvec_in) ? zero(T) : dvec_in[right_pad_idx]
    end
    sync_threads()
    conv_value = zero(T)
    for j in 1:mask_width
        conv_value += in_tile[threadIdx().x+j-1] * mask[j]
    end
    dvec_out[i] = conv_value
    nothing
end
