not_implemented() = throw(error("Not implemented!"))

get_value(::Val{T}) where {T} = T
get_value(v) = v

# convenience function so that I am not dealing with off-by-one indexes in my head all the time
view_length(xs::AbstractArray, base, width) = @view xs[base:base+width-1]

# using CUDA

# TODO: functions for querying device properties


