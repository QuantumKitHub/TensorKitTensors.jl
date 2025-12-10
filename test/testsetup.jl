module TensorKitTensorsTestSetup

export test_operator, operator_sum, swap_2sites, expanded_eigenvalues

using Test
using TensorKit
using TensorKit: scalartype
using LinearAlgebra: eigvals

function operator_sum(O::AbstractTensorMap; L::Int = 4)
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
        O1::AbstractTensorMap, O2::AbstractTensorMap; L::Int = 4,
        isapproxkwargs...
    )
    eigenvals1 = expanded_eigenvalues(O1; L)
    eigenvals2 = expanded_eigenvalues(O2; L)
    return @test isapprox(eigenvals1, eigenvals2; isapproxkwargs...)
end

function expanded_eigenvalues(O1::AbstractTensorMap; L::Int = 4)
    H = operator_sum(O1; L)
    eigenvals = mapreduce(vcat, pairs(eigvals(H))) do (c, vals)
        return repeat(vals, dim(c))
    end
    return sort!(eigenvals; by = real)
end

end
