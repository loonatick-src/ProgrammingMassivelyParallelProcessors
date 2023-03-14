function merge_sequential(v1::V, v2::V) where {V <: AbstractVector}
    m = length(v1); n = length(v2)
    v_merge = similar(v1, m + n)
    i = j = k = 1
    while i <= m && j <= n
        if v1[i] <= v2[j]
            v_merge[k] = v1[i]
            i+=1;
        else
            v_merge[k] = v2[j]
            j+=1
        end
        k+=1
    end
    if i == m+1
        while j <= n
            v_merge[k] = v2[j]
            j+=1;k+=1
        end
    else
        while i <= m
            v_merge[k] = v1[i]
            i+=1;k+=1
        end
    end
    v_merge
end
