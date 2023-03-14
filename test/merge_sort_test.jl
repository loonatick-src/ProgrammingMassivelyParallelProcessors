using ProgrammingMassivelyParallelProcessors

N = 1024
m = 333
n = N - m
v1 = rand(Int, m)
v2 = rand(Int, n)

sorted_base = sort(unsorted)
# sorted_merge = merge_sequential(
