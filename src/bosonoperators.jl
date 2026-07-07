module BosonOperators

using TensorKit
using LinearAlgebra: I
import ..TensorKitTensors: symmetrize, desymmetrize, @operator

export boson_space, basis_transform
export b_plus, b_min, b_num
export b_plus_b_plus, b_plus_b_min, b_min_b_plus, b_min_b_min
export b_hopping
export b⁻, b⁺, n
export b⁺b⁺, b⁻b⁺, b⁺b⁻, b⁻b⁻
export b_hop

"""
    boson_space([symmetry::Type{<:Sector}]; cutoff::Integer)

The local Hilbert space for a (truncated) bosonic system with a given symmetry and cutoff.
Available symmetries are `Trivial` or `U1Irrep`.
"""
boson_space(::Type{Trivial}; cutoff::Integer) = ComplexSpace(cutoff + 1)
boson_space(::Type{U1Irrep}; cutoff::Integer) = U1Space(n => 1 for n in 0:cutoff)
function boson_space(symmetry::Type{<:Sector}; kwargs...)
    throw(ArgumentError("invalid symmetry `$symmetry`"))
end

"""
    basis_transform(symmetry::Type{<:Sector}; cutoff::Integer)

Return the unitary basis transformation that maps the occupation-number basis
``\\{|0⟩, |1⟩, …, |\\mathrm{cutoff}⟩\\}`` of `boson_space(Trivial; cutoff)` onto the basis
of `boson_space(symmetry; cutoff)`, as a `TensorMap` from `boson_space(Trivial; cutoff)` to
`desymmetrize(boson_space(symmetry; cutoff))`, as required by
[`symmetrize`](@ref TensorKitTensors.symmetrize).

For `U1Irrep`, the boson number is used as the ``U(1)`` charge, and the charge sectors are
ordered as `0:cutoff`. This coincides with the occupation-number basis, such that the
transformation is the identity.

The transformations have exact integer entries and are therefore returned with integer
scalar type, such that they promote to any scalar type without loss of precision.
"""
function basis_transform(symmetry::Type{<:Sector}; cutoff::Integer)
    V = desymmetrize(boson_space(symmetry; cutoff))
    return TensorMap(Matrix{Int}(I, cutoff + 1, cutoff + 1), V ← boson_space(Trivial; cutoff))
end

# Symmetrize a boson operator through its basis transformation
_symmetrize_operator(O::AbstractTensorMap, symmetry::Type{<:Sector}; kwargs...) =
    symmetrize(O, basis_transform(symmetry; kwargs...), boson_space(symmetry; kwargs...))

# Single-site operators
# ---------------------
"""
    b_min([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁻([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic annihilation operator, with a maximum of `cutoff` bosons per site.
"""
@operator b⁻ b_min(::Type{<:Number}, ::Type{<:Sector}; cutoff)
function b_min(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    pspace = boson_space(Trivial; cutoff)
    b⁻ = zeros(elt, pspace ← pspace)
    for i in 1:cutoff
        b⁻[i, i + 1] = sqrt(i)
    end
    return b⁻
end

"""
    b_plus([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁺([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic creation operator, with a maximum of `cutoff` bosons per site.
"""
@operator b⁺ b_plus(::Type{<:Number}, ::Type{<:Sector}; cutoff)
function b_plus(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    pspace = boson_space(Trivial; cutoff)
    b⁺ = zeros(elt, pspace ← pspace)
    for i in 1:cutoff
        b⁺[i + 1, i] = sqrt(i)
    end
    return b⁺
end

"""
    b_num([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    n([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic number operator, with a maximum of `cutoff` bosons per site.
"""
@operator n b_num(::Type{<:Number}, ::Type{<:Sector}; cutoff)
function b_num(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    pspace = boson_space(Trivial; cutoff)
    n = zeros(elt, pspace ← pspace)
    for i in 0:cutoff
        n[i + 1, i + 1] = i
    end
    return n
end

# Two site operators
# ------------------
"""
    b_plus_b_plus([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁺b⁺([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic pair-creation operator, with a maximum of `cutoff` bosons per site.
"""
@operator b⁺b⁺ b_plus_b_plus(::Type{<:Number}, ::Type{<:Sector}; cutoff)
function b_plus_b_plus(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    b⁺ = b_plus(elt, Trivial; cutoff)
    return b⁺ ⊗ b⁺
end

"""
    b_plus_b_min([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁺b⁻([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic left-hopping operator, with a maximum of `cutoff` bosons per site.
"""
@operator b⁺b⁻ b_plus_b_min(::Type{<:Number}, ::Type{<:Sector}; cutoff)
function b_plus_b_min(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    b⁺ = b_plus(elt, Trivial; cutoff)
    b⁻ = b_min(elt, Trivial; cutoff)
    return b⁺ ⊗ b⁻
end

"""
    b_min_b_plus([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁻b⁺([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic right-hopping operator, with a maximum of `cutoff` bosons per site.
"""
@operator b⁻b⁺ b_min_b_plus(::Type{<:Number}, ::Type{<:Sector}; cutoff)
function b_min_b_plus(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    b⁺ = b_plus(elt, Trivial; cutoff)
    b⁻ = b_min(elt, Trivial; cutoff)
    return b⁻ ⊗ b⁺
end

"""
    b_min_b_min([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁻b⁻([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic pair-annihilation operator, with a maximum of `cutoff` bosons per site.
"""
@operator b⁻b⁻ b_min_b_min(::Type{<:Number}, ::Type{<:Sector}; cutoff)
function b_min_b_min(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    b⁻ = b_min(elt, Trivial; cutoff)
    return b⁻ ⊗ b⁻
end

"""
    b_hopping([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b_hop([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

Return the two-body operator that describes a particle that hops between the first and the second site.
"""
@operator b_hop b_hopping(::Type{<:Number}, ::Type{<:Sector}; cutoff)
function b_hopping(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    return b_plus_b_min(elt, Trivial; cutoff) + b_min_b_plus(elt, Trivial; cutoff)
end

end
