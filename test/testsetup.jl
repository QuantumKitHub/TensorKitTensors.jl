module TensorKitTensorsTestSetup

export test_operator, test_operator_dense, test_spin_algebra
export operator_sum, swap_2sites, expanded_eigenvalues, levicivita

using Test
using TensorKit
using TensorKitTensors: desymmetrize
using LinearAlgebra: eigvals, tr

const default_x = 0.361 + 0.729im
const default_L = 4

# su(2) structure constants εᵢⱼₖ, shared by the spin-algebra tests
const levicivita = let ε = zeros(Int, 3, 3, 3)
    for i in 1:3
        ε[mod1(i, 3), mod1(i + 1, 3), mod1(i + 2, 3)] = 1
        ε[mod1(i, 3), mod1(i - 1, 3), mod1(i - 2, 3)] = -1
    end
    ε
end

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

"""
    test_operator(O1, O2; x, L, isapproxkwargs...)

Compare two operators through the spectrum of the many-body Hamiltonian `O + x * O'`
summed over an `L`-site chain. This is basis-independent, so it happily compares a
symmetric operator with its trivial counterpart in a different basis ordering.

The spectrum is invariant under unitary conjugation (and transposition), so this test
cannot distinguish an operator from a unitarily-rotated version of itself. Use
[`test_operator_dense`](@ref) for an element-wise comparison that catches those.
"""
function test_operator(
        O1::AbstractTensorMap, O2::AbstractTensorMap;
        x::Number = default_x, L::Int = default_L, isapproxkwargs...
    )
    eigenvals1 = round_and_sort(expanded_eigenvalues(O1 + x * O1'; L))
    eigenvals2 = round_and_sort(expanded_eigenvalues(O2 + x * O2'; L))
    return @test isapprox(eigenvals1, eigenvals2; isapproxkwargs...)
end

"""
    test_operator_dense(O_sym, O_triv, U; isapproxkwargs...)
    test_operator_dense(O_sym, O_triv, Us::Tuple; isapproxkwargs...)

Compare a symmetric operator `O_sym` with its trivial counterpart `O_triv` element-wise in
a common dense basis, given the basis transformation(s) that map the trivial basis onto the
symmetric one (a module's `basis_transform`).

Both operators are brought to their dense form (see `desymmetrize`) and `O_triv` is rotated
per leg, so the test asserts `desymmetrize(O_sym) ≈ Uⁿ * desymmetrize(O_triv) * Uⁿ'` with
`Uⁿ = U₁ ⊗ ⋯ ⊗ Uₙ`. A single `U` is applied to every leg; passing a tuple `Us` allows a
site-dependent transformation, as needed for the staggered η-pairing gauge of the Hubbard
model under `SU2Irrep` particle symmetry.

Unlike [`test_operator`](@ref), which only compares spectra, this catches transposes, sign
errors, and gauge rotations. It is also far cheaper: no chain Hamiltonian, no diagonalization.
"""
function test_operator_dense(
        O_sym::AbstractTensorMap, O_triv::AbstractTensorMap, U::AbstractTensorMap;
        isapproxkwargs...
    )
    return test_operator_dense(
        O_sym, O_triv, ntuple(Returns(U), numout(O_triv)); isapproxkwargs...
    )
end
function test_operator_dense(
        O_sym::AbstractTensorMap, O_triv::AbstractTensorMap, Us::Tuple;
        isapproxkwargs...
    )
    Uⁿ = reduce(⊗, Us)
    return @test isapprox(
        desymmetrize(O_sym), Uⁿ * desymmetrize(O_triv) * Uⁿ'; isapproxkwargs...
    )
end

"""
    test_spin_algebra(Sx, Sy, Sz; spin = 1 // 2)

Check that `(Sx, Sy, Sz)` form a spin-`spin` representation of su(2): each component is
hermitian, the operators are normalized as `∑ᵢ tr(Sᵢ²) / (2s + 1) = s(s + 1)`, and they
satisfy the commutation relations `[Sᵢ, Sⱼ] = i εᵢⱼₖ Sₖ`.
"""
function test_spin_algebra(Sx, Sy, Sz; spin = 1 // 2)
    Svec = (Sx, Sy, Sz)
    for s in Svec
        @test s' ≈ s
    end
    @test sum(tr(Svec[i]^2) for i in 1:3) / (2spin + 1) ≈ spin * (spin + 1)
    for i in 1:3, j in 1:3
        @test Svec[i] * Svec[j] - Svec[j] * Svec[i] ≈
            sum(im * levicivita[i, j, k] * Svec[k] for k in 1:3)
    end
    return nothing
end

function expanded_eigenvalues(O::AbstractTensorMap; L::Int = default_L)
    H = operator_sum(O; L)
    eigenvals = mapreduce(vcat, pairs(eigvals(H))) do (c, vals)
        return repeat(vals, dim(c))
    end
    return sort!(eigenvals; by = real)
end

end
