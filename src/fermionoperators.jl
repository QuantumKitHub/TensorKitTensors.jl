module FermionOperators

using TensorKit

export fermion_space
export f_num
export f_plus_f_min, f_min_f_plus, f_plus_f_plus, f_min_f_min
export f_hopping
export n
export f⁺f⁻, f⁻f⁺, f⁺f⁺, f⁻f⁻
export f_hop

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
    f_num([elt::Type{<:Number}=ComplexF64])
    n([elt::Type{<:Number}=ComplexF64])

Return the one-body operator that counts the nunber of particles.
""" f_num
function f_num(T::Type{<:Number}=ComplexF64)
    t = single_site_operator(T)
    block(t, fℤ₂(1)) .= one(T)
    return t
end
const n = f_num

# Two site operators
# ------------------
function two_site_operator(T::Type{<:Number}=ComplexF64)
    V = fermion_space()
    return zeros(T, V ⊗ V ← V ⊗ V)
end

@doc """
    f_plus_f_min([elt::Type{<:Number}=ComplexF64])
    f⁺f⁻([elt::Type{<:Number}=ComplexF64])

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
""" f_plus_f_min
function f_plus_f_min(T::Type{<:Number}=ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))] .= 1
    return t
end
const f⁺f⁻ = f_plus_f_min

@doc """
    f_min_f_plus([elt::Type{<:Number}=ComplexF64])
    f⁻f⁺([elt::Type{<:Number}=ComplexF64])

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
""" f_min_f_plus
function f_min_f_plus(T::Type{<:Number}=ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(0), I(1), dual(I(1)), dual(I(0)))] .= -1
    return t
end
const f⁻f⁺ = f_min_f_plus

@doc """
    f_plus_f_plus([elt::Type{<:Number}=ComplexF64])
    f⁺f⁺([elt::Type{<:Number}=ComplexF64])

Return the two-body operator that creates a particle at the first and at the second site.
""" f_plus_f_plus
function f_plus_f_plus(T::Type{<:Number}=ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(1), I(1), dual(I(0)), dual(I(0)))] .= 1
    return t
end
const f⁺f⁺ = f_plus_f_plus

@doc """
    f_min_f_min([elt::Type{<:Number}=ComplexF64])
    f⁻f⁻([elt::Type{<:Number}=ComplexF64])

Return the two-body operator that annihilates a particle at the first and at the second site.
""" f_min_f_min
function f_min_f_min(T::Type{<:Number}=ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))] .= 1
    return t
end
const f⁻f⁻ = f_min_f_min

@doc """
    f_hopping([elt::Type{<:Number}=ComplexF64])
    f_hop([elt::Type{<:Number}=ComplexF64])

Return the two-body operator that describes a particle that hops between the first and the second site.
""" f_hopping
function f_hopping(elt::Type{<:Number}=ComplexF64)
    return f_plus_f_min(elt) - f_min_f_plus(elt)
end
const f_hop = f_hopping

end
