module FermionOperators

using TensorKit
using LinearAlgebra: I
import ..TensorKitTensors: symmetrize, _restrict_scalartype

export fermion_space, basis_transform
export f_num
export f_plus_f_min, f_min_f_plus, f_plus_f_plus, f_min_f_min
export f_hopping
export n
export f⁺f⁻, f⁻f⁺, f⁺f⁺, f⁻f⁻
export f_hop

"""
    fermion_space([symmetry::Type{<:Sector}])

Return the local hilbert space for a model of spinless fermions.
Available symmetries are `Trivial` or `U1Irrep`.
"""
fermion_space(::Type{Trivial}) = Vect[fℤ₂](0 => 1, 1 => 1)
fermion_space(::Type{U1Irrep}) = Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 1)
fermion_space() = fermion_space(Trivial)
fermion_space(symmetry::Type{<:Sector}) = throw(ArgumentError("invalid symmetry `$symmetry`"))

"""
    basis_transform(symmetry::Type{<:Sector})

Return the unitary basis transformation that maps the basis ``\\{|0⟩, |1⟩\\}`` of
`fermion_space(Trivial)` onto the basis of `fermion_space(symmetry)`, as required by
[`symmetrize`](@ref TensorKitTensors.symmetrize).

Note that even the `Trivial` fermionic space is graded by the fermion parity `fℤ₂`. For
`U1Irrep`, the particle number is additionally used as a ``U(1)`` charge, which refines the
grading without reordering the basis, such that the transformation is the identity.
"""
basis_transform(::Type{Trivial}) = Matrix{Float64}(I, 2, 2)
basis_transform(::Type{U1Irrep}) = Matrix{Float64}(I, 2, 2)

# Single-site operators
# ---------------------
function single_site_operator(T::Type{<:Number}, symmetry::Type{<:Sector} = Trivial)
    V = fermion_space(symmetry)
    return zeros(T, V ← V)
end

@doc """
    f_num([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    n([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the one-body operator that counts the nunber of particles.
""" f_num
function f_num(T::Type{<:Number}, ::Type{Trivial})
    t = single_site_operator(T, Trivial)
    block(t, fℤ₂(1)) .= one(T)
    return t
end
const n = f_num

# Two site operators
# ------------------
function two_site_operator(T::Type{<:Number}, symmetry::Type{<:Sector} = Trivial)
    V = fermion_space(symmetry)
    return zeros(T, V ⊗ V ← V ⊗ V)
end

@doc """
    f_plus_f_min([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f⁺f⁻([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
""" f_plus_f_min
function f_plus_f_min(T::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(T, Trivial)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))] .= 1
    return t
end
const f⁺f⁻ = f_plus_f_min

@doc """
    f_min_f_plus([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f⁻f⁺([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
""" f_min_f_plus
function f_min_f_plus(T::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(T, Trivial)
    I = sectortype(t)
    t[(I(0), I(1), dual(I(1)), dual(I(0)))] .= -1
    return t
end
const f⁻f⁺ = f_min_f_plus

@doc """
    f_plus_f_plus([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f⁺f⁺([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that creates a particle at the first and at the second site. It only has `Trivial` symmetry.
""" f_plus_f_plus
function f_plus_f_plus(T::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(T, Trivial)
    I = sectortype(t)
    t[(I(1), I(1), dual(I(0)), dual(I(0)))] .= 1
    return t
end
const f⁺f⁺ = f_plus_f_plus

@doc """
    f_min_f_min([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f⁻f⁻([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that annihilates a particle at the first and at the second site. It only has `Trivial` symmetry.
""" f_min_f_min
function f_min_f_min(T::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(T, Trivial)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))] .= -1
    return t
end
const f⁻f⁻ = f_min_f_min

@doc """
    f_hopping([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f_hop([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that describes a particle that hops between the first and the second site.
""" f_hopping
function f_hopping(elt::Type{<:Number}, ::Type{Trivial})
    return f_plus_f_min(elt, Trivial) - f_min_f_plus(elt, Trivial)
end
const f_hop = f_hopping

# Symmetric operators and default arguments
# -----------------------------------------
# The symmetric operators are automatically generated from their `Trivial` counterparts
# through `symmetrize` and `basis_transform`. Operators that are incompatible with a given
# symmetry throw an `ArgumentError`.
for opname in (:f_num, :f_plus_f_min, :f_min_f_plus, :f_plus_f_plus, :f_min_f_min, :f_hopping)
    @eval begin
        $opname() = $opname(ComplexF64, Trivial)
        $opname(elt::Type{<:Number}) = $opname(elt, Trivial)
        $opname(symmetry::Type{<:Sector}) = $opname(ComplexF64, symmetry)
        function $opname(elt::Type{<:Number}, symmetry::Type{<:Sector})
            O = $opname(complex(elt), Trivial)
            U = basis_transform(symmetry)
            O′ = symmetrize(O, U, fermion_space(symmetry); name = $(string(opname)))
            return _restrict_scalartype(elt, O′; name = $(string(opname)))
        end
    end
end

end
