module BosonOperators

using TensorKit
using LinearAlgebra: diagind

export boson_space
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

# Single-site operators
# ---------------------
@doc """
    b_min([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁻([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic annihilation operator, with a maximum of `cutoff` bosons per site.
""" b_min
b_min(; kwargs...) = b_min(ComplexF64, Trivial; kwargs...)
b_min(elt::Type{<:Number}; kwargs...) = b_min(elt, Trivial; kwargs...)
b_min(symm::Type{<:Sector}; kwargs...) = b_min(ComplexF64, symm; kwargs...)
function b_min(elt::Type{<:Number}, symmetry::Type{<:Sector}; cutoff::Integer)
    if symmetry === Trivial
        pspace = boson_space(Trivial; cutoff)
        b⁻ = zeros(elt, pspace ← pspace)
        for i in 1:cutoff
            b⁻[i, i + 1] = sqrt(i)
        end
        return b⁻
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
end

const b⁻ = b_min

@doc """
    b_plus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic creation operator, with a maximum of `cutoff` bosons per site.
""" b_plus
b_plus(; kwargs...) = b_plus(ComplexF64, Trivial; kwargs...)
b_plus(elt::Type{<:Number}; kwargs...) = b_plus(elt, Trivial; kwargs...)
b_plus(symm::Type{<:Sector}; kwargs...) = b_plus(ComplexF64, symm; kwargs...)
function b_plus(elt::Type{<:Number}, symmetry::Type{<:Sector}; cutoff::Integer)
    if symmetry === Trivial
        pspace = boson_space(Trivial; cutoff)
        b⁺ = zeros(elt, pspace ← pspace)
        for i in 1:cutoff
            b⁺[i + 1, i] = sqrt(i)
        end
        return b⁺
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
end

const b⁺ = b_plus

@doc """
    b_num([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    n([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic number operator, with a maximum of `cutoff` bosons per site.
""" b_num
b_num(; kwargs...) = b_num(ComplexF64, Trivial; kwargs...)
b_num(elt::Type{<:Number}; kwargs...) = b_num(elt, Trivial; kwargs...)
b_num(symm::Type{<:Sector}; kwargs...) = b_num(ComplexF64, symm; kwargs...)
function b_num(elt::Type{<:Number}, symmetry::Type{<:Sector}; cutoff::Integer)
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

const n = b_num

# Two site operators
# ------------------
@doc """
    b_plus_b_plus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁺b⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic pair-creation operator, with a maximum of `cutoff` bosons per site.
""" b_plus_b_plus
b_plus_b_plus(; kwargs...) = b_plus_b_plus(ComplexF64, Trivial; kwargs...)
b_plus_b_plus(elt::Type{<:Number}; kwargs...) = b_plus_b_plus(elt, Trivial; kwargs...)
b_plus_b_plus(symm::Type{<:Sector}; kwargs...) = b_plus_b_plus(ComplexF64, symm; kwargs...)
function b_plus_b_plus(elt::Type{<:Number}, symmetry::Type{<:Sector}; cutoff::Integer)
    if symmetry === Trivial
        b⁺ = b_plus(elt, Trivial; cutoff)
        return b⁺ ⊗ b⁺
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
end

const b⁺b⁺ = b_plus_b_plus

@doc """
    b_plus_b_min([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁺b⁻([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic left-hopping operator, with a maximum of `cutoff` bosons per site.
""" b_plus_b_min
b_plus_b_min(; kwargs...) = b_plus_b_min(ComplexF64, Trivial; kwargs...)
b_plus_b_min(elt::Type{<:Number}; kwargs...) = b_plus_b_min(elt, Trivial; kwargs...)
b_plus_b_min(symm::Type{<:Sector}; kwargs...) = b_plus_b_min(ComplexF64, symm; kwargs...)
function b_plus_b_min(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    b⁺ = b_plus(elt, Trivial; cutoff)
    b⁻ = b_min(elt, Trivial; cutoff)
    return b⁺ ⊗ b⁻
end
function b_plus_b_min(elt::Type{<:Number}, ::Type{U1Irrep}; cutoff::Integer)
    pspace = boson_space(U1Irrep; cutoff)
    b⁺b⁻ = zeros(elt, pspace ⊗ pspace ← pspace ⊗ pspace)
    for (f1, f2) in fusiontrees(b⁺b⁻)
        c_out, c_in = f1.uncoupled, f2.uncoupled
        if c_in[1].charge + 1 == c_out[1].charge &&
           c_in[2].charge - 1 == c_out[2].charge
            b⁺b⁻[f1, f2] .= sqrt(c_out[1].charge) * sqrt(c_in[2].charge)
        end
    end
    return b⁺b⁻
end

const b⁺b⁻ = b_plus_b_min

@doc """
    b_min_b_plus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁻b⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic right-hopping operator, with a maximum of `cutoff` bosons per site.
""" b_min_b_plus
b_min_b_plus(; kwargs...) = b_min_b_plus(ComplexF64, Trivial; kwargs...)
b_min_b_plus(elt::Type{<:Number}; kwargs...) = b_min_b_plus(elt, Trivial; kwargs...)
b_min_b_plus(symm::Type{<:Sector}; kwargs...) = b_min_b_plus(ComplexF64, symm; kwargs...)
function b_min_b_plus(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer)
    b⁺ = b_plus(elt, Trivial; cutoff)
    b⁻ = b_min(elt, Trivial; cutoff)
    return b⁻ ⊗ b⁺
end
function b_min_b_plus(elt::Type{<:Number}, ::Type{U1Irrep}; cutoff::Integer)
    pspace = boson_space(U1Irrep; cutoff)
    b⁻b⁺ = zeros(elt, pspace ⊗ pspace ← pspace ⊗ pspace)
    for (f1, f2) in fusiontrees(b⁻b⁺)
        c_out, c_in = f1.uncoupled, f2.uncoupled
        if c_in[1].charge - 1 == c_out[1].charge &&
           c_in[2].charge + 1 == c_out[2].charge
            b⁻b⁺[f1, f2] .= sqrt(c_in[1].charge) * sqrt(c_out[2].charge)
        end
    end
    return b⁻b⁺
end

const b⁻b⁺ = b_min_b_plus

@doc """
    b_min_b_min([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)
    b⁻b⁻([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff::Integer)

The truncated bosonic pair-annihilation operator, with a maximum of `cutoff` bosons per site.
""" b_min_b_min
b_min_b_min(; kwargs...) = b_min_b_min(ComplexF64, Trivial; kwargs...)
b_min_b_min(elt::Type{<:Number}; kwargs...) = b_min_b_min(elt, Trivial; kwargs...)
b_min_b_min(symm::Type{<:Sector}; kwargs...) = b_min_b_min(ComplexF64, symm; kwargs...)
function b_min_b_min(elt::Type{<:Number}, symmetry::Type{<:Sector}; cutoff::Integer)
    if symmetry === Trivial
        b⁻ = b_min(elt, Trivial; cutoff)
        return b⁻ ⊗ b⁻
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
end

const b⁻b⁻ = b_min_b_min

@doc """
    b_hopping([eltype::Type{<:Number}])
    b_hop([eltype::Type{<:Number}])

Return the two-body operator that describes a particle that hops between the first and the second site.
""" b_hop
b_hopping(; kwargs...) = b_hopping(ComplexF64, Trivial; kwargs...)
function b_hopping(elt::Type{<:Number}, symmetry::Type{<:Sector}; cutoff::Integer)
    return b_plus_b_min(elt, symmetry; cutoff) + b_min_b_plus(elt, symmetry; cutoff)
end
const b_hop = b_hopping

end
