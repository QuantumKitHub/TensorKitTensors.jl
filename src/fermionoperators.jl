#===========================================================================================
    Spinless fermions
===========================================================================================#

module FermionOperators

using TensorKit

export c_plus, c_min
export c_plus_c_min, c_min_c_plus, c_plus_c_plus, c_min_c_min
export c_num
export c⁺, c⁻
export c⁺c⁻, c⁻c⁺, c⁺c⁺, c⁻c⁻
export n

"""
    c_num([elt::Type{<:Number}=ComplexF64])

Fermionic number operator.
"""
function c_num(T::Type{<:Number}=ComplexF64)
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    n = zeros(T, pspace ← pspace)
    block(n, fℤ₂(1)) .= one(T)
    return n
end
const n = c_num

# Two site operators
# ------------------
function two_site_operator(T::Type{<:Number}=ComplexF64)
    V = Vect[fℤ₂](0 => 1, 1 => 1)
    return zeros(T, V ⊗ V ← V ⊗ V)
end

function c_plus_c_min(T=ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))] .= 1
    return t
end
const c⁺c⁻ = c_plus_c_min

function c_min_c_plus(T=ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(0), I(1), dual(I(1)), dual(I(0)))] .= 1
    return t
end
const c⁻c⁺ = c_min_c_plus

function c_plus_c_plus(T=ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(1), I(1), dual(I(0)), dual(I(0)))] .= 1
    return t
end
const c⁺c⁺ = c_plus_c_plus

function c_min_c_min(T=ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))] .= 1
    return t
end
const c⁻c⁻ = c_min_c_min

end
