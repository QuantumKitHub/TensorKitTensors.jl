module FermionOperators

using TensorKit
using LinearAlgebra: I
import ..TensorKitTensors: symmetrize, desymmetrize, @operator

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

# Symmetrize a fermion operator through its basis transformation
_symmetrize_operator(O::AbstractTensorMap, symmetry::Type{<:Sector}) =
    symmetrize(O, basis_transform(symmetry), fermion_space(symmetry))

# Single-site operators
# ---------------------
function single_site_operator(T::Type{<:Number}, symmetry::Type{<:Sector})
    V = fermion_space(symmetry)
    return zeros(T, V ← V)
end

"""
    f_num([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    n([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the one-body operator that counts the nunber of particles.
"""
@operator n function f_num(T::Type{<:Number}, ::Type{Trivial})
    t = single_site_operator(T, Trivial)
    block(t, fℤ₂(1)) .= one(T)
    return t
end

# Two site operators
# ------------------
function two_site_operator(T::Type{<:Number}, symmetry::Type{<:Sector})
    V = fermion_space(symmetry)
    return zeros(T, V ⊗ V ← V ⊗ V)
end

"""
    f_plus_f_min([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f⁺f⁻([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
"""
@operator f⁺f⁻ function f_plus_f_min(T::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(T, Trivial)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))] .= 1
    return t
end

"""
    f_min_f_plus([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f⁻f⁺([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
"""
@operator f⁻f⁺ function f_min_f_plus(T::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(T, Trivial)
    I = sectortype(t)
    t[(I(0), I(1), dual(I(1)), dual(I(0)))] .= -1
    return t
end

"""
    f_plus_f_plus([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f⁺f⁺([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that creates a particle at the first and at the second site. It only has `Trivial` symmetry.
"""
@operator f⁺f⁺ function f_plus_f_plus(T::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(T, Trivial)
    I = sectortype(t)
    t[(I(1), I(1), dual(I(0)), dual(I(0)))] .= 1
    return t
end

"""
    f_min_f_min([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f⁻f⁻([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that annihilates a particle at the first and at the second site. It only has `Trivial` symmetry.
"""
@operator f⁻f⁻ function f_min_f_min(T::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(T, Trivial)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))] .= -1
    return t
end

"""
    f_hopping([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f_hop([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that describes a particle that hops between the first and the second site.
"""
@operator f_hop function f_hopping(elt::Type{<:Number}, ::Type{Trivial})
    return f_plus_f_min(elt, Trivial) - f_min_f_plus(elt, Trivial)
end

end
