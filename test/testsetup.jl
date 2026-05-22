module TensorKitTensorsTestSetup

export test_operator, operator_sum, swap_2sites, expanded_eigenvalues

using Test
using TensorKit
using TensorKit: scalartype
using LinearAlgebra: eigvals

const default_x = 0.361 + 0.729im
const default_L = 4

# utility function to sort a complex vector
# rounding to eliminate real/imag part very close to 0
function round_and_sort(evs::Vector{<:Number}; digits = 12)
    evs2 = round.(evs; digits)
    return sort!(evs2; by = z -> (real(z), imag(z)))
end

function operator_sum(O::AbstractTensorMap; L::Int = default_L)
    I = id(space(O, 1))
    n = numin(O)
    return sum(1:(L - n + 1)) do i
        return reduce(⊗, insert!(collect(Any, fill(I, L - n)), i, O))
    end
end

function swap_2sites(op::AbstractTensorMap{T, S, 2, 2}) where {T, S}
    return permute(op, ((2, 1), (4, 3)))
end

function test_operator(
        O1::AbstractTensorMap, O2::AbstractTensorMap;
        x::Number = default_x, L::Int = default_L, isapproxkwargs...
    )
    eigenvals1 = round_and_sort(expanded_eigenvalues(O1 + x * O1'; L))
    eigenvals2 = round_and_sort(expanded_eigenvalues(O2 + x * O2'; L))
    return @test isapprox(eigenvals1, eigenvals2; isapproxkwargs...)
end

function expanded_eigenvalues(O::AbstractTensorMap; L::Int = default_L)
    H = operator_sum(O; L)
    eigenvals = mapreduce(vcat, pairs(eigvals(H))) do (c, vals)
        return repeat(vals, dim(c))
    end
    return sort!(eigenvals; by = real)
end

end
