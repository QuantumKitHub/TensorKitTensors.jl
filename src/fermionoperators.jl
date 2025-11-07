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
    fermion_space([symmetry::Type{<:Sector}])

Return the local hilbert space for a model of spinless fermions.
Available symmetries are `Trivial` or `U1Irrep`.
"""
fermion_space(::Type{Trivial}) = Vect[fℤ₂](0 => 1, 1 => 1)
fermion_space(::Type{U1Irrep}) = Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 1)
fermion_space() = fermion_space(Trivial)

# Single-site operators
# ---------------------
function single_site_operator(T::Type{<:Number}, symmetry::Type{<:Sector} = Trivial)
    V = fermion_space(symmetry)
    return zeros(T, V ← V)
end

@doc """
    f_num([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    n([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the one-body operator that counts the nunber of particles.
""" f_num
function f_num(T::Type{<:Number}, (::Type{Trivial}))
    t = single_site_operator(T, Trivial)
    block(t, fℤ₂(1)) .= one(T)
    return t
end
function f_num(T::Type{<:Number}, (::Type{U1Irrep}))
    t = single_site_operator(T, U1Irrep)
    S = sectortype(t)
    block(t, S(1, 1)) .= one(T)
    return t
end
f_num(elt::Type{<:Number}) = f_num(elt, Trivial)
f_num(sym::Type{<:Sector}) = f_num(ComplexF64, sym)
const n = f_num

# Two site operators
# ------------------
function two_site_operator(T::Type{<:Number}, symmetry::Type{<:Sector} = Trivial)
    V = fermion_space(symmetry)
    return zeros(T, V ⊗ V ← V ⊗ V)
end

@doc """
    f_plus_f_min([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f⁺f⁻([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
""" f_plus_f_min
function f_plus_f_min(T::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(T, Trivial)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))] .= 1
    return t
end
function f_plus_f_min(T::Type{<:Number}, ::Type{U1Irrep})
    t = two_site_operator(T, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1), I(0, 0), dual(I(0, 0)), dual(I(1, 1)))] .= 1
    return t
end
f_plus_f_min(elt::Type{<:Number}) = f_plus_f_min(elt, Trivial)
f_plus_f_min(sym::Type{<:Sector}) = f_plus_f_min(ComplexF64, sym)
const f⁺f⁻ = f_plus_f_min

@doc """
    f_min_f_plus([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f⁻f⁺([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
""" f_min_f_plus
function f_min_f_plus(T::Type{<:Number}, ::Type{Trivial})
    t = two_site_operator(T, Trivial)
    I = sectortype(t)
    t[(I(0), I(1), dual(I(1)), dual(I(0)))] .= -1
    return t
end
function f_min_f_plus(T::Type{<:Number}, ::Type{U1Irrep})
    t = two_site_operator(T, U1Irrep)
    I = sectortype(t)
    t[(I(0, 0), I(1, 1), dual(I(1, 1)), dual(I(0, 0)))] .= -1
    return t
end
f_min_f_plus(elt::Type{<:Number}) = f_min_f_plus(elt, Trivial)
f_min_f_plus(sym::Type{<:Sector}) = f_min_f_plus(ComplexF64, sym)
const f⁻f⁺ = f_min_f_plus

@doc """
    f_plus_f_plus([elt::Type{<:Number}=ComplexF64])
    f⁺f⁺([elt::Type{<:Number}=ComplexF64])

Return the two-body operator that creates a particle at the first and at the second site.
""" f_plus_f_plus
function f_plus_f_plus(T::Type{<:Number} = ComplexF64)
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
function f_min_f_min(T::Type{<:Number} = ComplexF64)
    t = two_site_operator(T)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))] .= -1
    return t
end
const f⁻f⁻ = f_min_f_min

@doc """
    f_hopping([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])
    f_hop([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial])

Return the two-body operator that describes a particle that hops between the first and the second site.
""" f_hopping
function f_hopping(elt::Type{<:Number}, symmetry::Type{<:Sector})
    return f_plus_f_min(elt, symmetry) - f_min_f_plus(elt, symmetry)
end
f_hopping(elt::Type{<:Number}) = f_hopping(elt, Trivial)
f_hopping(sym::Type{<:Sector}) = f_hopping(ComplexF64, sym)
const f_hop = f_hopping

end
