module FermionOperators

using TensorKit
using LinearAlgebra: I
import ..TensorKitTensors: symmetrize, desymmetrize, @operator

export fermion_space, basis_transform, symmetrize_operator
export f_num
export f_plus_f_min, f_min_f_plus, f_plus_f_plus, f_min_f_min
export f_hopping
export n
export f⁺f⁻, f⁻f⁺, f⁺f⁺, f⁻f⁻
export f_hop

"""
    fermion_space([symmetry::Type{<:Sector}])

The local Hilbert space for a single spinless fermionic mode, spanned by the empty and the
occupied state ``\\{|0⟩, |1⟩\\}``.

| Symmetry | Space |
|---|---|
| `Trivial` | `Vect[fℤ₂](0 => 1, 1 => 1)` |
| `U1Irrep` | `Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 1)`, using the particle number as charge |

The space is always graded by the fermion parity `fℤ₂`, even for `Trivial` symmetry: the
grading is what makes TensorKit insert the anticommutation signs when operators on different
sites are contracted, and `Trivial` refers only to the absence of an additional symmetry.
Since a `TensorMap` on a graded space only has parity-preserving blocks, the parity-odd
single-site operators ``f^+`` and ``f^-`` are not representable at all, and only their
parity-even two-site combinations are provided.
"""
fermion_space(::Type{Trivial}) = Vect[fℤ₂](0 => 1, 1 => 1)
fermion_space(::Type{U1Irrep}) = Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 1)
fermion_space() = fermion_space(Trivial)
fermion_space(symmetry::Type{<:Sector}) = throw(ArgumentError("invalid symmetry `$symmetry`"))

"""
    basis_transform(symmetry::Type{<:Sector})

Return the unitary basis transformation that maps the basis ``\\{|0⟩, |1⟩\\}`` of
`fermion_space(Trivial)` onto the basis of `fermion_space(symmetry)`, as a `TensorMap` from
`desymmetrize(fermion_space(Trivial))` to `desymmetrize(fermion_space(symmetry))`, as
required by [`symmetrize`](@ref TensorKitTensors.symmetrize). Note that both sides are
purely bosonic `ComplexSpace`s, since a `TensorMap` cannot mix different gradings.

Even the `Trivial` fermionic space is graded by the fermion parity `fℤ₂`. For `U1Irrep`,
the particle number is additionally used as a ``U(1)`` charge, which refines the grading
without reordering the basis, such that the transformation is the identity. It is returned
with integer scalar type, such that it promotes to any scalar type without loss of
precision.
"""
function basis_transform(symmetry::Type{<:Sector})
    V = desymmetrize(fermion_space(symmetry))
    return TensorMap(Matrix{Int}(I, 2, 2), V ← desymmetrize(fermion_space(Trivial)))
end

"""
    symmetrize_operator(O::AbstractTensorMap, symmetry::Type{<:Sector}; tol=nothing)

Symmetrize a spinless-fermion operator defined on `fermion_space(Trivial)` through the basis transformation for `symmetry`.
The input space must retain the mandatory fermion-parity grading.
"""
function symmetrize_operator(
        O::AbstractTensorMap, symmetry::Type{<:Sector}; tol = nothing
    )
    return symmetrize(O, basis_transform(symmetry), fermion_space(symmetry); tol)
end

# Single-site operators
# ---------------------
function single_site_operator(elt::Type{<:Number}, symmetry::Type{<:Sector})
    V = fermion_space(symmetry)
    return zeros(elt, V ← V)
end

"""
    f_num([elt::Type{<:Number}], [symmetry::Type{<:Sector}])
    n([elt::Type{<:Number}], [symmetry::Type{<:Sector}])

Return the one-body operator that counts the number of particles, ``n = f^+ f^-``, which is
diagonal with eigenvalues ``0`` (empty) and ``1`` (occupied).

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator n function f_num(elt::Type{<:Number}, ::Type{Trivial})
    t = single_site_operator(elt, Trivial)
    block(t, fℤ₂(1)) .= one(elt)
    return t
end

# Two site operators
# ------------------
function two_site_operator(elt::Type{<:Number}, symmetry::Type{<:Sector})
    V = fermion_space(symmetry)
    return zeros(elt, V ⊗ V ← V ⊗ V)
end

"""
    f_plus_f_min([elt::Type{<:Number}], [symmetry::Type{<:Sector}])
    f⁺f⁻([elt::Type{<:Number}], [symmetry::Type{<:Sector}])

Return the two-body operator ``f^+_1 f^-_2`` that creates a particle at the first site and
annihilates a particle at the second.

Supported symmetries: `Trivial`, `U1Irrep`.

See also [`f_min_f_plus`](@ref) (``f^-_1 f^+_2 = -(f^+_1 f^-_2)^†``) and
[`f_hopping`](@ref).
"""
@operator f⁺f⁻ function f_plus_f_min(elt::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(elt, Trivial)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))] .= 1
    return t
end

"""
    f_min_f_plus([elt::Type{<:Number}], [symmetry::Type{<:Sector}])
    f⁻f⁺([elt::Type{<:Number}], [symmetry::Type{<:Sector}])

Return the two-body operator ``f^-_1 f^+_2`` that annihilates a particle at the first site and
creates a particle at the second. It picks up the anticommutation sign that comes with
reordering the two fermionic operators, i.e. ``f^-_1 f^+_2 = -(f^+_1 f^-_2)^†``.

Supported symmetries: `Trivial`, `U1Irrep`.

See also [`f_plus_f_min`](@ref) and [`f_hopping`](@ref).
"""
@operator f⁻f⁺ function f_min_f_plus(elt::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(elt, Trivial)
    I = sectortype(t)
    t[(I(0), I(1), dual(I(1)), dual(I(0)))] .= -1
    return t
end

"""
    f_plus_f_plus([elt::Type{<:Number}], [symmetry::Type{<:Sector}])
    f⁺f⁺([elt::Type{<:Number}], [symmetry::Type{<:Sector}])

Return the two-body operator ``f^+_1 f^+_2`` that creates a particle at the first and at the
second site, with ``|1,1⟩ = f^+_1 f^+_2 |0,0⟩`` fixing the sign of the reference state. It
changes the number of particles by two, so it preserves the fermion parity but not the
particle number.

Supported symmetries: `Trivial`.

See also [`f_min_f_min`](@ref) (``f^-_1 f^-_2 = -(f^+_1 f^+_2)^†``).
"""
@operator f⁺f⁺ function f_plus_f_plus(elt::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(elt, Trivial)
    I = sectortype(t)
    t[(I(1), I(1), dual(I(0)), dual(I(0)))] .= 1
    return t
end

"""
    f_min_f_min([elt::Type{<:Number}], [symmetry::Type{<:Sector}])
    f⁻f⁻([elt::Type{<:Number}], [symmetry::Type{<:Sector}])

Return the two-body operator ``f^-_1 f^-_2`` that annihilates a particle at the first and at
the second site. It picks up the anticommutation sign relative to
``|1,1⟩ = f^+_1 f^+_2 |0,0⟩``, i.e. ``f^-_1 f^-_2 = -(f^+_1 f^+_2)^†``. It changes the number
of particles by two, so it preserves the fermion parity but not the particle number.

Supported symmetries: `Trivial`.

See also [`f_plus_f_plus`](@ref).
"""
@operator f⁻f⁻ function f_min_f_min(elt::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(elt, Trivial)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))] .= -1
    return t
end

"""
    f_hopping([elt::Type{<:Number}], [symmetry::Type{<:Sector}])
    f_hop([elt::Type{<:Number}], [symmetry::Type{<:Sector}])

Return the two-body operator that describes a particle that hops between the first and the
second site,
```math
f_\\mathrm{hop} = f^+_1 f^-_2 - f^-_1 f^+_2 = f^+_1 f^-_2 + (f^+_1 f^-_2)^†,
```
which is hermitian. Note the minus sign, which is what makes this combination hermitian here:
``f^-_1 f^+_2`` already carries the anticommutation sign, whereas the corresponding bosonic
hopping operator is a sum.

Supported symmetries: `Trivial`, `U1Irrep`.

See also [`f_plus_f_min`](@ref) and [`f_min_f_plus`](@ref).
"""
@operator f_hop function f_hopping(elt::Type{<:Number}, ::Type{Trivial})
    return f_plus_f_min(elt, Trivial) - f_min_f_plus(elt, Trivial)
end

end
