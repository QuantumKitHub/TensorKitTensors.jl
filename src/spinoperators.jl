module SpinOperators

using TensorKit
using LinearAlgebra: diagind

export spin_space, casimir
export S_x, S_y, S_z, S_plus, S_min
export S_x_S_x, S_y_S_y, S_z_S_z, S_plus_S_min, S_min_S_plus, S_exchange
export σˣ, σʸ, σᶻ, σ⁺, σ⁻
export Sˣ, Sʸ, Sᶻ, S⁺, S⁻
export SˣSˣ, SʸSʸ, SᶻSᶻ, S⁺S⁻, S⁻S⁺, SS

"""
    spin_space([symmetry::Type{<:Sector}]; spin=1 // 2)

The local Hilbert space for a spin system with a given symmetry and spin.
Available symmetries are `Trivial`, `Z2Irrep`, `U1Irrep` and `SU2Irrep`.
"""
spin_space(::Type{Trivial}=Trivial; spin=1 // 2) = ComplexSpace(Int(2 * spin + 1))
function spin_space(::Type{Z2Irrep}; spin=1 // 2)
    spin == 1 // 2 || throw(ArgumentError("Z2 symmetry only implemented for spin 1//2"))
    return Z2Space(0 => 1, 1 => 1)
end
spin_space(::Type{U1Irrep}; spin=1 // 2) = U1Space(i => 1 for i in (-spin):spin)
spin_space(::Type{SU2Irrep}; spin=1 // 2) = SU2Space(spin => 1)

# Pauli matrices
# --------------
function _pauliterm(spin, i, j)
    1 <= i <= 2 * spin + 1 || return 0.0
    1 <= j <= 2 * spin + 1 || return 0.0
    return sqrt((spin + 1) * (i + j - 1) - i * j) / 2.0
end
function _pauliterm(spin, i::U1Irrep, j::U1Irrep)
    -spin <= i.charge <= spin || return 0.0
    -spin <= j.charge <= spin || return 0.0
    return sqrt((spin + 1) * (i.charge + j.charge + 2 * spin + 1) -
                (i.charge + spin + 1) * (j.charge + spin + 1)) / 2.0
end
casimir(spin::Real) = spin * (spin + 1)
casimir(c::SU2Irrep) = casimir(c.j)

"""
    spinmatrices(spin [, eltype])

The spinmatrices according to [Wikipedia](https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins).
"""
function spinmatrices(spin::Union{Rational{Int},Int}, elt=ComplexF64)
    N = Int(2 * spin)

    Sx = zeros(elt, N + 1, N + 1)
    Sy = zeros(complex(elt), N + 1, N + 1)
    Sz = zeros(elt, N + 1, N + 1)

    for row in 1:(N + 1)
        for col in 1:(N + 1)
            term = _pauliterm(spin, row, col)

            if (row + 1 == col)
                Sx[row, col] += term
                Sy[row, col] -= 1im * term
            end

            if (row == col + 1)
                Sx[row, col] += term
                Sy[row, col] += 1im * term
            end

            if (row == col)
                Sz[row, col] += spin + 1 - row
            end
        end
    end
    return Sx, Sy, Sz, one(Sx)
end

# Single-site operators
# ---------------------
@doc """
    S_x([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    Sˣ([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin operator along the x-axis.
See also [`σˣ`](@ref)
""" S_x
S_x(; kwargs...) = S_x(ComplexF64, Trivial; kwargs...)
S_x(elt::Type{<:Number}; kwargs...) = S_x(elt, Trivial; kwargs...)
S_x(symm::Type{<:Sector}; kwargs...) = S_x(ComplexF64, symm; kwargs...)
function S_x(elt::Type{<:Number}, symmetry::Type{<:Sector}; spin=1 // 2)
    pspace = spin_space(symmetry; spin)
    if symmetry === Trivial
        S_x_mat, = spinmatrices(spin, elt)
        X = TensorMap(S_x_mat, pspace ← pspace)
    elseif symmetry === Z2Irrep
        X = zeros(elt, pspace ← pspace)
        for (c, b) in blocks(X)
            b .= c.n == 1 ? -one(elt) / 2 : one(elt) / 2
        end
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
    return X
end
const Sˣ = S_x

"""Pauli x operator."""
σˣ(args...; kwargs...) = 2 * S_x(args...; kwargs...)

@doc """
    S_y([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    Sʸ([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin operator along the y-axis.
See also [`σʸ`](@ref)
""" S_y
S_y(; kwargs...) = S_y(ComplexF64, Trivial; kwargs...)
S_y(elt::Type{<:Complex}; kwargs...) = S_y(elt, Trivial; kwargs...)
S_y(symm::Type{<:Sector}; kwargs...) = S_y(ComplexF64, symm; kwargs...)
function S_y(elt::Type{<:Complex}, symmetry::Type{<:Sector}; spin=1 // 2)
    pspace = spin_space(symmetry; spin)
    if symmetry === Trivial
        _, S_y_mat, _ = spinmatrices(spin, elt)
        Y = TensorMap(S_y_mat, pspace ← pspace)
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
    return Y
end
const Sʸ = S_y

"""Pauli y operator."""
σʸ(args...; kwargs...) = 2 * S_y(args...; kwargs...)

@doc """
    S_z([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    Sᶻ([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin operator along the z-axis.
See also [`σᶻ`](@ref)
""" S_z
S_z(; kwargs...) = S_z(ComplexF64, Trivial; kwargs...)
S_z(elt::Type{<:Number}; kwargs...) = S_z(elt, Trivial; kwargs...)
S_z(symm::Type{<:Sector}; kwargs...) = S_z(ComplexF64, symm; kwargs...)
function S_z(elt::Type{<:Number}, symmetry::Type{<:Sector}; spin=1 // 2)
    pspace = spin_space(symmetry; spin)
    if symmetry === Trivial
        _, _, S_z_mat = spinmatrices(spin, elt)
        Z = TensorMap(S_z_mat, pspace ← pspace)
    elseif symmetry === U1Irrep
        Z = zeros(elt, pspace ← pspace)
        for (c, b) in blocks(Z)
            b .= c.charge
        end
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
    return Z
end
const Sᶻ = S_z

"""Pauli z operator."""
σᶻ(args...; kwargs...) = 2 * S_z(args...; kwargs...)

@doc """
    S_plus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    S⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin raising operator.
See also [`σ⁺`](@ref)
""" S_plus
S_plus(; kwargs...) = S_plus(ComplexF64, Trivial; kwargs...)
S_plus(elt::Type{<:Number}; kwargs...) = S_plus(elt, Trivial; kwargs...)
S_plus(symm::Type{<:Sector}; kwargs...) = S_plus(ComplexF64, symm; kwargs...)
function S_plus(elt::Type{<:Number}, symmetry::Type{<:Sector}; spin=1 // 2)
    if symmetry === Trivial
        S⁺ = S_x(elt, Trivial; spin) + 1im * S_y(complex(elt), Trivial; spin)
        if elt <: Real
            S⁺ = real(S⁺)
        end
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
    return S⁺
end
const S⁺ = S_plus

"""Pauli raising operator."""
σ⁺(args...; kwargs...) = 2 * S_plus(args...; kwargs...)

@doc """
    S_min([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    S⁻([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin lowering operator.
See also [`σ⁻`](@ref)
""" S_min
S_min(; kwargs...) = S_min(ComplexF64, Trivial; kwargs...)
S_min(elt::Type{<:Number}; kwargs...) = S_min(elt, Trivial; kwargs...)
S_min(symm::Type{<:Sector}; kwargs...) = S_min(ComplexF64, symm; kwargs...)
function S_min(elt::Type{<:Number}, symmetry::Type{<:Sector}; spin=1 // 2)
    if symmetry === Trivial
        S⁻ = S_x(elt, Trivial; spin) - 1im * S_y(complex(elt), Trivial; spin)
        if elt <: Real
            S⁻ = real(S⁻)
        end
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
    return S⁻
end
const S⁻ = S_min

"""Pauli minus operator."""
σ⁻(args...; kwargs...) = 2 * S_min(args...; kwargs...)

# Two site operators
# ------------------
@doc """
    S_x_S_x([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    SˣSˣ([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin x exchange operator.
""" S_x_S_x
S_x_S_x(; kwargs...) = S_x_S_x(ComplexF64, Trivial; kwargs...)
S_x_S_x(elt::Type{<:Number}; kwargs...) = S_x_S_x(elt, Trivial; kwargs...)
S_x_S_x(symm::Type{<:Sector}; kwargs...) = S_x_S_x(ComplexF64, symm; kwargs...)
function S_x_S_x(elt::Type{<:Number}, symmetry::Type{<:Sector}; spin=1 // 2)
    if symmetry === Trivial || symmetry === Z2Irrep
        XX = S_x(elt, symmetry; spin) ⊗ S_x(elt, symmetry; spin)
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
    return XX
end
const SˣSˣ = S_x_S_x

@doc """
    S_y_S_y([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    SʸSʸ([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin y exchange operator.
""" S_y_S_y
S_y_S_y(; kwargs...) = S_y_S_y(ComplexF64, Trivial; kwargs...)
S_y_S_y(elt::Type{<:Complex}; kwargs...) = S_y_S_y(elt, Trivial; kwargs...)
S_y_S_y(symm::Type{<:Sector}; kwargs...) = S_y_S_y(ComplexF64, symm; kwargs...)
function S_y_S_y(elt::Type{<:Complex}, symmetry::Type{<:Sector}; spin=1 // 2)
    if symmetry === Trivial
        YY = S_y(elt, Trivial; spin) ⊗ S_y(elt, Trivial; spin)
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
    return YY
end
const SʸSʸ = S_y_S_y

@doc """
    S_z_S_z([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    SᶻSᶻ([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin z exchange operator.
""" S_z_S_z
S_z_S_z(; kwargs...) = S_z_S_z(ComplexF64, Trivial; kwargs...)
S_z_S_z(elt::Type{<:Number}; kwargs...) = S_z_S_z(elt, Trivial; kwargs...)
S_z_S_z(symm::Type{<:Sector}; kwargs...) = S_z_S_z(ComplexF64, symm; kwargs...)
function S_z_S_z(elt::Type{<:Number}, symmetry::Type{<:Sector}; spin=1 // 2)
    if symmetry === Trivial || symmetry === U1Irrep
        ZZ = S_z(elt, symmetry; spin) ⊗ S_z(elt, symmetry; spin)
    elseif symmetry === Z2Irrep
        pspace = spin_space(Z2Irrep; spin)
        ZZ = zeros(elt, pspace ⊗ pspace ← pspace ⊗ pspace)
        for (f₁, f₂) in fusiontrees(ZZ)
            if f₁.uncoupled[1] != f₂.uncoupled[1] && f₁.uncoupled[2] != f₂.uncoupled[2]
                ZZ[f₁, f₂] .= one(elt) / 4
            end
        end
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
    return ZZ
end
const SᶻSᶻ = S_z_S_z

@doc """
    S_plus_S_min([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    S⁺S⁻([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin raising-lowering operator.
""" S_plus_S_min
S_plus_S_min(; kwargs...) = S_plus_S_min(ComplexF64, Trivial; kwargs...)
S_plus_S_min(elt::Type{<:Number}; kwargs...) = S_plus_S_min(elt, Trivial; kwargs...)
S_plus_S_min(symm::Type{<:Sector}; kwargs...) = S_plus_S_min(ComplexF64, symm; kwargs...)
function S_plus_S_min(elt::Type{<:Number}, symmetry::Type{<:Sector}; spin=1 // 2)
    if symmetry === Trivial
        S⁺S⁻ = S_plus(elt, symmetry; spin) ⊗ S_min(elt, symmetry; spin)
    elseif symmetry === U1Irrep
        pspace = spin_space(U1Irrep; spin)
        S⁺S⁻ = zeros(elt, pspace ⊗ pspace ← pspace ⊗ pspace)
        for (f₁, f₂) in fusiontrees(S⁺S⁻)
            if f₁.uncoupled[1].charge == f₂.uncoupled[1].charge + 1 &&
               f₁.uncoupled[2].charge == f₂.uncoupled[2].charge - 1
                m₁, m₂ = getproperty.(f₂.uncoupled, :charge)
                S⁺S⁻[f₁, f₂] .= sqrt(casimir(spin) - m₁ * (m₁ + 1)) *
                                sqrt(casimir(spin) - m₂ * (m₂ - 1))
            end
        end
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
    return S⁺S⁻
end
const S⁺S⁻ = S_plus_S_min

@doc """
    S_min_S_plus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    S⁻S⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin lowering-raising operator.
""" S_min_S_plus
S_min_S_plus(; kwargs...) = S_min_S_plus(ComplexF64, Trivial; kwargs...)
S_min_S_plus(elt::Type{<:Number}; kwargs...) = S_min_S_plus(elt, Trivial; kwargs...)
S_min_S_plus(symm::Type{<:Sector}; kwargs...) = S_min_S_plus(ComplexF64, symm; kwargs...)
function S_min_S_plus(elt::Type{<:Number}, symmetry::Type{<:Sector}; spin=1 // 2)
    if symmetry === Trivial
        S⁻S⁺ = S_min(elt, symmetry; spin) ⊗ S_plus(elt, symmetry; spin)
    elseif symmetry === U1Irrep
        pspace = spin_space(U1Irrep; spin)
        S⁻S⁺ = zeros(elt, pspace ⊗ pspace ← pspace ⊗ pspace)
        for (f₁, f₂) in fusiontrees(S⁻S⁺)
            if f₁.uncoupled[1].charge == f₂.uncoupled[1].charge - 1 &&
               f₁.uncoupled[2].charge == f₂.uncoupled[2].charge + 1
                m₁, m₂ = getproperty.(f₂.uncoupled, :charge)
                S⁻S⁺[f₁, f₂] .= sqrt(casimir(spin) - m₁ * (m₁ - 1)) *
                                sqrt(casimir(spin) - m₂ * (m₂ + 1))
            end
        end
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
    return S⁻S⁺
end
const S⁻S⁺ = S_min_S_plus

@doc """
    S_exchange([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    SS([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin exchange operator.
""" S_exchange
S_exchange(; kwargs...) = S_exchange(ComplexF64, Trivial; kwargs...)
S_exchange(elt::Type{<:Number}; kwargs...) = S_exchange(elt, Trivial; kwargs...)
function S_exchange(symmetry::Type{<:Sector}; kwargs...)
    return S_exchange(ComplexF64, symmetry; kwargs...)
end
function S_exchange(elt::Type{<:Number}, symmetry::Type{<:Sector}; spin=1 // 2)
    if symmetry === Trivial || symmetry === U1Irrep
        SS = (S_plus_S_min(elt, symmetry; spin) + S_min_S_plus(elt, symmetry; spin)) / 2 +
             S_z_S_z(elt, symmetry; spin)
    elseif symmetry === SU2Irrep
        pspace = spin_space(SU2Irrep; spin)
        SS = zeros(elt, pspace ⊗ pspace ← pspace ⊗ pspace)
        for (c, b) in blocks(SS)
            @inbounds for i in diagind(b)
                b[i] = casimir(c) / 2 - casimir(spin)
            end
        end
    else
        throw(ArgumentError("invalid symmetry `$symmetry`"))
    end
    return SS
end
const SS = S_exchange

end
