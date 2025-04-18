module FermionOperators

using TensorKit

export fermion_space
export c_num
export c_plus_c_min, c_min_c_plus, c_plus_c_plus, c_min_c_min
export n
export c⁺c⁻, c⁻c⁺, c⁺c⁺, c⁻c⁻

"""
    fermion_space()

Return the local hilbert space for a model of spinless fermions.
"""
fermion_space() = Vect[fℤ₂](0 => 1, 1 => 1)

# Single-site operators
# ---------------------
function single_site_operator(T)
    V = fermion_space()
    return zeros(T, V ← V)
end

@doc """
    c_num([elt::Type{<:Number}=ComplexF64])
    n([elt::Type{<:Number}=ComplexF64])

Return the one-body operator that counts the nunber of particles.
""" c_num
function c_num(T::Type{<:Number}=ComplexF64)
    t = single_site_operator(T)
    block(t, fℤ₂(1)) .= one(T)
    return t
end
const n = c_num

# Two site operators
# ------------------
function two_site_operator(T::Type{<:Number}=ComplexF64)
    V = fermion_space()
    return zeros(T, V ⊗ V ← V ⊗ V)
end

@doc """
    c_plus_c_min([elt::Type{<:Number}=ComplexF64])
    c⁺c⁻([elt::Type{<:Number}=ComplexF64])

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
""" c_plus_c_min
function c_plus_c_min(T::Type{<:Number}=ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))] .= 1
    return t
end
const c⁺c⁻ = c_plus_c_min

@doc """
    c_min_c_plus([elt::Type{<:Number}=ComplexF64])
    c⁻c⁺([elt::Type{<:Number}=ComplexF64])

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
""" c_min_c_plus
function c_min_c_plus(T::Type{<:Number}=ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(0), I(1), dual(I(1)), dual(I(0)))] .= -1
    return t
end
const c⁻c⁺ = c_min_c_plus

@doc """
    c_plus_c_plus([elt::Type{<:Number}=ComplexF64])
    c⁺c⁺([elt::Type{<:Number}=ComplexF64])

Return the two-body operator that creates a particle at the first and at the second site.
""" c_plus_c_plus
function c_plus_c_plus(T::Type{<:Number}=ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(1), I(1), dual(I(0)), dual(I(0)))] .= 1
    return t
end
const c⁺c⁺ = c_plus_c_plus

@doc """
    c_min_c_min([elt::Type{<:Number}=ComplexF64])
    c⁻c⁻([elt::Type{<:Number}=ComplexF64])

Return the two-body operator that annihilates a particle at the first and at the second site.
""" c_min_c_min
function c_min_c_min(T::Type{<:Number}=ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))] .= 1
    return t
end
const c⁻c⁻ = c_min_c_min

end
