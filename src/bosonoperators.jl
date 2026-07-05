module BosonOperators

using TensorKit
using LinearAlgebra: I
import ..TensorKitTensors: symmetrize, _restrict_scalartype

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
    basis_transform([elt::Type{<:Number}], symmetry::Type{<:Sector}; cutoff::Integer)

Return the unitary basis transformation that maps the occupation-number basis
``\\{|0⟩, |1⟩, …, |\\mathrm{cutoff}⟩\\}`` of `boson_space(Trivial; cutoff)` onto the basis
of `boson_space(symmetry; cutoff)`, as required by [`symmetrize`](@ref TensorKitTensors.symmetrize).

For `U1Irrep`, the boson number is used as the ``U(1)`` charge, and the charge sectors are
ordered as `0:cutoff`. This coincides with the occupation-number basis, such that the
transformation is the identity.

The transformations have exact integer entries and are therefore returned as integer
matrices, irrespective of `elt`, such that they promote to any scalar type without loss of
precision.
"""
function basis_transform(symmetry::Type{<:Sector}; kwargs...)
    return basis_transform(Float64, symmetry; kwargs...)
end
function basis_transform(::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    return Matrix{Int}(I, cutoff + 1, cutoff + 1)
end
function basis_transform(::Type{<:Number}, ::Type{U1Irrep}; cutoff::Integer)
    return Matrix{Int}(I, cutoff + 1, cutoff + 1)
end

# Single-site operators
# ---------------------
@doc """
    b_min([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁻([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic annihilation operator, with a maximum of `cutoff` bosons per site.
""" b_min
function b_min(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    pspace = boson_space(Trivial; cutoff)
    b⁻ = zeros(elt, pspace ← pspace)
    for i in 1:cutoff
        b⁻[i, i + 1] = sqrt(i)
    end
    return b⁻
end
const b⁻ = b_min

@doc """
    b_plus([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁺([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic creation operator, with a maximum of `cutoff` bosons per site.
""" b_plus
function b_plus(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    pspace = boson_space(Trivial; cutoff)
    b⁺ = zeros(elt, pspace ← pspace)
    for i in 1:cutoff
        b⁺[i + 1, i] = sqrt(i)
    end
    return b⁺
end
const b⁺ = b_plus

@doc """
    b_num([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    n([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic number operator, with a maximum of `cutoff` bosons per site.
""" b_num
function b_num(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    pspace = boson_space(Trivial; cutoff)
    n = zeros(elt, pspace ← pspace)
    for i in 0:cutoff
        n[i + 1, i + 1] = i
    end
    return n
end
const n = b_num

# Two site operators
# ------------------
@doc """
    b_plus_b_plus([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁺b⁺([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic pair-creation operator, with a maximum of `cutoff` bosons per site.
""" b_plus_b_plus
function b_plus_b_plus(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    b⁺ = b_plus(elt, Trivial; cutoff)
    return b⁺ ⊗ b⁺
end
const b⁺b⁺ = b_plus_b_plus

@doc """
    b_plus_b_min([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁺b⁻([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic left-hopping operator, with a maximum of `cutoff` bosons per site.
""" b_plus_b_min
function b_plus_b_min(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    b⁺ = b_plus(elt, Trivial; cutoff)
    b⁻ = b_min(elt, Trivial; cutoff)
    return b⁺ ⊗ b⁻
end
const b⁺b⁻ = b_plus_b_min

@doc """
    b_min_b_plus([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁻b⁺([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic right-hopping operator, with a maximum of `cutoff` bosons per site.
""" b_min_b_plus
function b_min_b_plus(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    b⁺ = b_plus(elt, Trivial; cutoff)
    b⁻ = b_min(elt, Trivial; cutoff)
    return b⁻ ⊗ b⁺
end
const b⁻b⁺ = b_min_b_plus

@doc """
    b_min_b_min([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁻b⁻([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic pair-annihilation operator, with a maximum of `cutoff` bosons per site.
""" b_min_b_min
function b_min_b_min(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    b⁻ = b_min(elt, Trivial; cutoff)
    return b⁻ ⊗ b⁻
end
const b⁻b⁻ = b_min_b_min

@doc """
    b_hopping([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b_hop([elt::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

Return the two-body operator that describes a particle that hops between the first and the second site.
""" b_hopping
function b_hopping(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    return b_plus_b_min(elt, Trivial; cutoff) + b_min_b_plus(elt, Trivial; cutoff)
end
const b_hop = b_hopping

# Symmetric operators and default arguments
# -----------------------------------------
# The symmetric operators are automatically generated from their `Trivial` counterparts
# through `symmetrize` and `basis_transform`. Operators that are incompatible with a given
# symmetry throw an `ArgumentError`.
for opname in
    (:b_min, :b_plus, :b_num, :b_plus_b_plus, :b_plus_b_min, :b_min_b_plus, :b_min_b_min, :b_hopping)
    @eval begin
        $opname(; kwargs...) = $opname(ComplexF64, Trivial; kwargs...)
        $opname(elt::Type{<:Number}; kwargs...) = $opname(elt, Trivial; kwargs...)
        $opname(symmetry::Type{<:Sector}; kwargs...) = $opname(ComplexF64, symmetry; kwargs...)
        function $opname(elt::Type{<:Number}, symmetry::Type{<:Sector}; cutoff::Integer)
            O = $opname(elt, Trivial; cutoff)
            U = basis_transform(elt, symmetry; cutoff)
            O′ = symmetrize(O, U, boson_space(symmetry; cutoff); name = $(string(opname)))
            return _restrict_scalartype(elt, O′; name = $(string(opname)))
        end
    end
end

end
