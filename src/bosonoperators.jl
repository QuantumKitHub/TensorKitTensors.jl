module BosonOperators

using Core: Argument
using TensorKit
using LinearAlgebra: diagind

export boson_space
export a_plus, a_min, number
export a_plusplus, a_plusmin, a_minplus, a_minmin
export a, a⁺, n
export a⁺a⁺, aa⁺, a⁺a, aa

"""
    boson_space([symmetry::Type{<:Sector}]; cutoff::Integer)

The local Hilbert space for a (truncated) bosonic system with a given symmetry and cutoff.
Available symmetries are `Trivial` or `U1Irrep`.
"""
boson_space(::Type{Trivial}; cutoff::Integer) = ComplexSpace(cutoff + 1)
boson_space(::Type{U1Irrep}; cutoff::Integer) = U1Space(n => 1 for n in 0:cutoff)

# Single-site operators
# ---------------------
@doc """
    a_min([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    a([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic annihilation operator, with a maximum of `cutoff` bosons per site.
""" a_min
a_min(; kwargs...) = a_min(ComplexF64, Trivial; kwargs...)
a_min(elt::Type{<:Number}; kwargs...) = a_min(elt, Trivial; kwargs...)
a_min(symm::Type{<:Sector}; kwargs...) = a_min(ComplexF64, symm; kwargs...)
function a_min(elt::Type{<:Number}, symmetry::Type{<:Sector}; cutoff::Integer)
    if symmetry === Trivial
        pspace = boson_space(Trivial; cutoff)
        a = zeros(elt, pspace ← pspace)
        for i in 1:cutoff
            a[i, i + 1] = sqrt(i)
        end
        return a
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
end

const a = a_min

@doc """
    a_plus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    a⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic creation operator, with a maximum of `cutoff` bosons per site.
""" a_plus
a_plus(; kwargs...) = a_plus(ComplexF64, Trivial; kwargs...)
a_plus(elt::Type{<:Number}; kwargs...) = a_plus(elt, Trivial; kwargs...)
a_plus(symm::Type{<:Sector}; kwargs...) = a_plus(ComplexF64, symm; kwargs...)
function a_plus(elt::Type{<:Number}, symmetry::Type{<:Sector}; cutoff::Integer)
    if symmetry === Trivial
        pspace = boson_space(Trivial; cutoff)
        a⁺ = zeros(elt, pspace ← pspace)
        for i in 1:cutoff
            a⁺[i + 1, i] = sqrt(i)
        end
        return a⁺
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
end

const a⁺ = a_plus

@doc """
    number([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    n([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic number operator, with a maximum of `cutoff` bosons per site.
""" number
number(; kwargs...) = number(ComplexF64, Trivial; kwargs...)
number(elt::Type{<:Number}; kwargs...) = number(elt, Trivial; kwargs...)
number(symm::Type{<:Sector}; kwargs...) = number(ComplexF64, symm; kwargs...)
function number(elt::Type{<:Number}, symmetry::Type{<:Sector}; cutoff::Integer)
    pspace = boson_space(symmetry; cutoff)
    n = zeros(elt, pspace ← pspace)
    if symmetry === Trivial
        for i in 0:cutoff
            n[i + 1, i + 1] = i
        end
    elseif symmetry === U1Irrep
        for (c, b) in blocks(n)
            b .= c.charge
        end
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
    return n
end

const n = number

# Two site operators
# ------------------
@doc """
    a_plusplus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    a⁺a⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic pair-creation operator, with a maximum of `cutoff` bosons per site.
""" a_plusplus
a_plusplus(; kwargs...) = a_plusplus(ComplexF64, Trivial; kwargs...)
a_plusplus(elt::Type{<:Number}; kwargs...) = a_plusplus(elt, Trivial; kwargs...)
a_plusplus(symm::Type{<:Sector}; kwargs...) = a_plusplus(ComplexF64, symm; kwargs...)
function a_plusplus(elt::Type{<:Number}, symmetry::Type{<:Sector}; cutoff::Integer)
    if symmetry === Trivial
        a⁺ = a_plus(elt, Trivial; cutoff)
        return a⁺ ⊗ a⁺
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
end

const a⁺a⁺ = a_plusplus

@doc """
    a_plusmin([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    a⁺a([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic left-hopping operator, with a maximum of `cutoff` bosons per site.
""" a_plusmin
a_plusmin(; kwargs...) = a_plusmin(ComplexF64, Trivial; kwargs...)
a_plusmin(elt::Type{<:Number}; kwargs...) = a_plusmin(elt, Trivial; kwargs...)
a_plusmin(symm::Type{<:Sector}; kwargs...) = a_plusmin(ComplexF64, symm; kwargs...)
function a_plusmin(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    a⁺ = a_plus(elt, Trivial; cutoff)
    a = a_min(elt, Trivial; cutoff)
    return a⁺ ⊗ a
end
function a_plusmin(elt::Type{<:Number}, ::Type{U1Irrep}; cutoff::Integer)
    pspace = boson_space(U1Irrep; cutoff)
    a⁺a = zeros(elt, pspace ⊗ pspace ← pspace ⊗ pspace)
    for (f1, f2) in fusiontrees(a⁺a)
        c_out, c_in = f1.uncoupled, f2.uncoupled
        if c_in[1].charge + 1 == c_out[1].charge &&
           c_in[2].charge - 1 == c_out[2].charge
            a⁺a[f1, f2] .= sqrt(c_out[1].charge) * sqrt(c_in[2].charge)
        end
    end
    return a⁺a
end

const a⁺a = a_plusmin

@doc """
    a_minplus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    aa⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic right-hopping operator, with a maximum of `cutoff` bosons per site.
""" a_minplus
a_minplus(; kwargs...) = a_minplus(ComplexF64, Trivial; kwargs...)
a_minplus(elt::Type{<:Number}; kwargs...) = a_minplus(elt, Trivial; kwargs...)
a_minplus(symm::Type{<:Sector}; kwargs...) = a_minplus(ComplexF64, symm; kwargs...)
function a_minplus(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    a⁺ = a_plus(elt, Trivial; cutoff)
    a = a_min(elt, Trivial; cutoff)
    return a ⊗ a⁺
end
function a_minplus(elt::Type{<:Number}, ::Type{U1Irrep}; cutoff::Integer)
    pspace = boson_space(U1Irrep; cutoff)
    aa⁺ = zeros(elt, pspace ⊗ pspace ← pspace ⊗ pspace)
    for (f1, f2) in fusiontrees(aa⁺)
        c_out, c_in = f1.uncoupled, f2.uncoupled
        if c_in[1].charge - 1 == c_out[1].charge &&
           c_in[2].charge + 1 == c_out[2].charge
            aa⁺[f1, f2] .= sqrt(c_in[1].charge) * sqrt(c_out[2].charge)
        end
    end
    return aa⁺
end

const aa⁺ = a_minplus

@doc """
    a_minmin([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    aa([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic pair-annihilation operator, with a maximum of `cutoff` bosons per site.
""" a_minmin
a_minmin(; kwargs...) = a_minmin(ComplexF64, Trivial; kwargs...)
a_minmin(elt::Type{<:Number}; kwargs...) = a_minmin(elt, Trivial; kwargs...)
a_minmin(symm::Type{<:Sector}; kwargs...) = a_minmin(ComplexF64, symm; kwargs...)
function a_minmin(elt::Type{<:Number}, symmetry::Type{<:Sector}; cutoff::Integer)
    if symmetry === Trivial
        a = a_min(elt, Trivial; cutoff)
        return a ⊗ a
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
end

const aa = a_minmin

end
