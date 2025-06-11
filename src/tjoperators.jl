module TJOperators

using TensorKit

export tj_space
export e_num, u_num, d_num, h_num
export S_x, S_y, S_z, S_plus, S_min
export u_plus_u_min, d_plus_d_min
export u_min_u_plus, d_min_d_plus
export u_min_d_min, d_min_u_min
export u_min_u_min, d_min_d_min
export u_plus_u_plus, d_plus_d_plus
export e_plus_e_min, e_min_e_plus, e_hopping
export singlet_min
export S_plus_S_min, S_min_S_plus, S_exchange

export nꜛ, nꜜ, nʰ, n
export Sˣ, Sʸ, Sᶻ, S⁺, S⁻
export u⁺u⁻, d⁺d⁻, u⁻u⁺, d⁻d⁺
export u⁻d⁻, d⁻u⁻
export u⁻u⁻, u⁺u⁺, d⁻d⁻, d⁺d⁺
export e⁺e⁻, e⁻e⁺, e_hop
export singlet⁻
export S⁻S⁺, S⁺S⁻

"""
    tj_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the local hilbert space for a t-J-type model with the given particle and spin symmetries.
The possible symmetries are 
- Particle number: `Trivial`, `U1Irrep`;
- Spin: `Trivial`, `U1Irrep`, `SU2Irrep`.

Setting `slave_fermion = true` switches to the slave-fermion basis. 

- basis states for `slave_fermion = false`: 
    |0⟩ = |vac⟩ (vacuum), |↑⟩ = (c↑)†|vac⟩, |↓⟩ = (c↓)†|vac⟩
- basis states for `slave_fermion = true`: (c_σ = h† b_σ; holon h is fermionic, spinon b_σ is bosonic): 
    |0⟩ = h†|vac⟩, |↑⟩ = (b↑)†|vac⟩, |↓⟩ = (b↓)†|vac⟩
"""
function tj_space(::Type{Trivial}=Trivial, ::Type{Trivial}=Trivial;
                  slave_fermion::Bool=false)
    return slave_fermion ? Vect[FermionParity](0 => 2, 1 => 1) :
           Vect[FermionParity](0 => 1, 1 => 2)
end
function tj_space(::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    return if slave_fermion
        Vect[FermionParity ⊠ U1Irrep]((1, 0) => 1, (0, 1 // 2) => 1, (0, -1 // 2) => 1)
    else
        Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
    end
end
function tj_space(::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    return slave_fermion ? Vect[FermionParity ⊠ SU2Irrep]((1, 0) => 1, (0, 1 // 2) => 1) :
           Vect[FermionParity ⊠ SU2Irrep]((0, 0) => 1, (1, 1 // 2) => 1)
end
function tj_space(::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    return if slave_fermion
        Vect[FermionParity ⊠ U1Irrep]((1, 0) => 1, (0, 1) => 2)
    else
        Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 2)
    end
end
function tj_space(::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    return if slave_fermion
        Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((1, 0, 0) => 1, (0, 1, 1 // 2) => 1,
                                                (0, 1, -1 // 2) => 1)
    else
        Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1,
                                                (1, 1, -1 // 2) => 1)
    end
end
function tj_space(::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    return if slave_fermion
        Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((1, 0, 0) => 1, (0, 1, 1 // 2) => 1)
    else
        Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1)
    end
end

# Single-site operators
# ---------------------
function single_site_operator(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                              spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    V = tj_space(particle_symmetry, spin_symmetry; slave_fermion)
    return zeros(elt, V ← V)
end

@doc """
    u_num(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    nꜛ(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the one-body operator that counts the number of spin-up electrons.
""" u_num
function u_num(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return u_num(ComplexF64, P, S; slave_fermion)
end
function u_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
               slave_fermion::Bool=false)
    t = single_site_operator(elt, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][1, 1] = 1
    return t
end
function u_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep};
               slave_fermion::Bool=false)
    t = single_site_operator(elt, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1 // 2), dual(I(b, 1 // 2)))][1, 1] = 1
    return t
end
function u_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep};
               slave_fermion::Bool=false)
    throw(ArgumentError("`u_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial};
               slave_fermion::Bool=false)
    t = single_site_operator(elt, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][1, 1] = 1
    return t
end
function u_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep};
               slave_fermion::Bool=false)
    t = single_site_operator(elt, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1, 1 // 2), dual(I(b, 1, 1 // 2)))] .= 1
    return t
end
function u_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep};
               slave_fermion::Bool=false)
    throw(ArgumentError("`u_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
const nꜛ = u_num

@doc """
    d_num(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    nꜜ(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body operator that counts the number of spin-down electrons.
""" d_num
function d_num(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return d_num(ComplexF64, P, S; slave_fermion)
end
function d_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
               slave_fermion::Bool=false)
    t = single_site_operator(elt, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][2, 2] = 1
    return t
end
function d_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep};
               slave_fermion::Bool=false)
    t = single_site_operator(elt, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, -1 // 2), dual(I(b, -1 // 2)))][1, 1] = 1
    return t
end
function d_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep};
               slave_fermion::Bool=false)
    throw(ArgumentError("`d_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial};
               slave_fermion::Bool=false)
    t = single_site_operator(elt, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][2, 2] = 1
    return t
end
function d_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep};
               slave_fermion::Bool=false)
    t = single_site_operator(elt, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1, -1 // 2), dual(I(b, 1, -1 // 2)))] .= 1
    return t
end
function d_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep};
               slave_fermion::Bool=false)
    throw(ArgumentError("`d_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
const nꜜ = d_num

@doc """
    e_num(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    n(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body operator that counts the number of particles.
""" e_num
function e_num(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_num(ComplexF64, P, S; slave_fermion)
end
function e_num(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
               spin_symmetry::Type{<:Sector};
               slave_fermion::Bool=false)
    return u_num(elt, particle_symmetry, spin_symmetry; slave_fermion) +
           d_num(elt, particle_symmetry, spin_symmetry; slave_fermion)
end
function e_num(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep};
               slave_fermion::Bool=false)
    t = single_site_operator(elt, Trivial, SU2Irrep; slave_fermion)
    I = sectortype(t)
    if slave_fermion
        block(t, I(0, 1 // 2))[1, 1] = 1
    else
        block(t, I(1, 1 // 2))[1, 1] = 1
    end
    return t
end
function e_num(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep};
               slave_fermion::Bool=false)
    t = single_site_operator(elt, U1Irrep, SU2Irrep; slave_fermion)
    I = sectortype(t)
    if slave_fermion
        block(t, I(0, 1, 1 // 2))[1, 1] = 1
    else
        block(t, I(1, 1, 1 // 2))[1, 1] = 1
    end
    return t
end
const n = e_num

@doc """
    h_num(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    nʰ(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body operator that counts the number of holes.
""" h_num
function h_num(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return h_num(ComplexF64, P, S; slave_fermion)
end
function h_num(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
               spin_symmetry::Type{<:Sector};
               slave_fermion::Bool=false)
    iden = TensorKit.id(tj_space(particle_symmetry, spin_symmetry; slave_fermion))
    return iden - e_num(elt, particle_symmetry, spin_symmetry; slave_fermion)
end
const nʰ = h_num

@doc """
    S_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    S⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the spin-plus operator (only defined for `Trivial` spin symmetry).
""" S_plus
function S_plus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_plus(ComplexF64, P, S; slave_fermion)
end
function S_plus(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
                slave_fermion::Bool=false)
    t = single_site_operator(elt, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][1, 2] = 1.0
    return t
end
function S_plus(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial};
                slave_fermion::Bool=false)
    t = single_site_operator(elt, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][1, 2] = 1.0
    return t
end
function S_plus(elt::Type{<:Number}, ::Type{<:Sector}, ::Type{U1Irrep};
                slave_fermion::Bool=false)
    throw(ArgumentError("`S_plus`, `S_min` are not symmetric under `U1Irrep` spin symmetry"))
end
function S_plus(elt::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep};
                slave_fermion::Bool=false)
    throw(ArgumentError("`S_plus`, `S_min` are not symmetric under `SU2Irrep` spin symmetry"))
end
const S⁺ = S_plus

@doc """
    S_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    S⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the spin-minus operator (only defined for `Trivial` spin symmetry).
""" S_min
function S_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_min(ComplexF64, P, S; slave_fermion)
end
function S_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
               spin_symmetry::Type{<:Sector};
               slave_fermion::Bool=false)
    return copy(adjoint(S_plus(elt, particle_symmetry, spin_symmetry; slave_fermion)))
end
const S⁻ = S_min

@doc """
    S_x(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    Sˣ(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body spin-1/2 x-operator on the electrons (only defined for `Trivial` spin symmetry).
""" S_x
function S_x(P::Type{<:Sector}=Trivial, S::Type{<:Sector}=Trivial;
             slave_fermion::Bool=false)
    return S_x(ComplexF64, P, S; slave_fermion)
end
function S_x(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
             spin_symmetry::Type{<:Sector};
             slave_fermion::Bool=false)
    return (S_plus(elt, particle_symmetry, spin_symmetry; slave_fermion)
            +
            S_min(elt, particle_symmetry, spin_symmetry; slave_fermion)) / 2
end
const Sˣ = S_x

@doc """
    S_y(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    Sʸ(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body spin-1/2 y-operator on the electrons (only defined for `Trivial` spin symmetry). 
""" S_y
function S_y(P::Type{<:Sector}=Trivial, S::Type{<:Sector}=Trivial;
             slave_fermion::Bool=false)
    return S_y(ComplexF64, P, S; slave_fermion)
end
function S_y(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
             spin_symmetry::Type{<:Sector};
             slave_fermion::Bool=false)
    return (S_plus(elt, particle_symmetry, spin_symmetry; slave_fermion)
            -
            S_min(elt, particle_symmetry, spin_symmetry; slave_fermion)) / (2im)
end
const Sʸ = S_y

@doc """
    S_z(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    Sᶻ(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body spin-1/2 z-operator on the electrons. 
""" S_z
function S_z(P::Type{<:Sector}=Trivial, S::Type{<:Sector}=Trivial;
             slave_fermion::Bool=false)
    return S_z(ComplexF64, P, S; slave_fermion)
end
function S_z(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
             spin_symmetry::Type{<:Sector};
             slave_fermion::Bool=false)
    return (u_num(elt, particle_symmetry, spin_symmetry; slave_fermion) -
            d_num(elt, particle_symmetry, spin_symmetry; slave_fermion)) / 2
end
const Sᶻ = S_z

# Two site operators
# ------------------
function two_site_operator(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                           spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    V = tj_space(particle_symmetry, spin_symmetry; slave_fermion)
    return zeros(elt, V ⊗ V ← V ⊗ V)
end

@doc """
    u_plus_u_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    u⁺u⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↑}, e_{2,↑}`` that creates a spin-up electron at the first site and annihilates a spin-up electron at the second.
The only nonzero matrix element corresponds to `|↑0⟩ <-- |0↑⟩`.
""" u_plus_u_min
function u_plus_u_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return u_plus_u_min(ComplexF64, P, S; slave_fermion)
end
function u_plus_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    #= The extra minus sign in slave-fermion basis:
    c†_{1,↑} c_{2,↑} |0↑⟩
    = h_1 b†_{1,↑} h†_2 b_{2,↑} h†_1 b†_{2,↑}|vac⟩
    = -b†_{1,↑} h†_2 h_1 h†_1 b_{2,↑} b†_{2,↑}|vac⟩
    = -b†_{1,↑} h†_2 |vac⟩
    = -|↑0⟩
    =#
    t[(I(b), I(h), dual(I(h)), dual(I(b)))][1, 1, 1, 1] = sgn * 1
    return t
end
function u_plus_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1 // 2), I(h, 0), dual(I(h, 0)), dual(I(b, 1 // 2)))] .= sgn * 1
    return t
end
function u_plus_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep};
                      slave_fermion::Bool=false)
    throw(ArgumentError("`u_plus_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_plus_u_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1), I(h, 0), dual(I(h, 0)), dual(I(b, 1)))][1, 1, 1, 1] = sgn * 1
    return t
end
function u_plus_u_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1, 1 // 2), I(h, 0, 0), dual(I(h, 0, 0)), dual(I(b, 1, 1 // 2)))] .= sgn * 1
    return t
end
function u_plus_u_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep};
                      slave_fermion::Bool=false)
    throw(ArgumentError("`u_plus_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
const u⁺u⁻ = u_plus_u_min

@doc """
    d_plus_d_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    d⁺d⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↓}, e_{2,↓}`` that creates a spin-down electron at the first site and annihilates a spin-down electron at the second.
The only nonzero matrix element corresponds to `|↓0⟩ <-- |0↓⟩`.
""" d_plus_d_min
function d_plus_d_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return d_plus_d_min(ComplexF64, P, S; slave_fermion)
end
function d_plus_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b), I(h), dual(I(h)), dual(I(b)))][2, 1, 1, 2] = sgn * 1
    return t
end
function d_plus_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt::Type{<:Number}, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, -1 // 2), I(h, 0), dual(I(h, 0)), dual(I(b, -1 // 2)))] .= sgn * 1
    return t
end
function d_plus_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep};
                      slave_fermion::Bool=false)
    throw(ArgumentError("`d_plus_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_plus_d_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1), I(h, 0), dual(I(h, 0)), dual(I(b, 1)))][2, 1, 1, 2] = sgn * 1
    return t
end
function d_plus_d_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1, -1 // 2), I(h, 0, 0), dual(I(h, 0, 0)), dual(I(b, 1, -1 // 2)))] .= sgn * 1
    return t
end
function d_plus_d_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep};
                      slave_fermion::Bool=false)
    throw(ArgumentError("`d_plus_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
const d⁺d⁻ = d_plus_d_min

@doc """
    u_min_u_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    u⁻u⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↑}, e†_{2,↑}`` that annihilates a spin-up electron at the first site and creates a spin-up electron at the second.
The only nonzero matrix element corresponds to `|0↑⟩ <-- |↑0⟩`.
""" u_min_u_plus
function u_min_u_plus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return u_min_u_plus(ComplexF64, P, S; slave_fermion)
end
function u_min_u_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                      spin_symmetry::Type{<:Sector};
                      slave_fermion::Bool=false)
    return -copy(adjoint(u_plus_u_min(elt, particle_symmetry, spin_symmetry; slave_fermion)))
end
const u⁻u⁺ = u_min_u_plus

@doc """
    d_min_d_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    d⁻d⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↓}, e†_{2,↓}`` that annihilates a spin-down electron at the first site and creates a spin-down electron at the second.
The only nonzero matrix element corresponds to `|0↓⟩ <-- |↓0⟩`.
""" d_min_d_plus
function d_min_d_plus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return d_min_d_plus(ComplexF64, P, S; slave_fermion)
end
function d_min_d_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                      spin_symmetry::Type{<:Sector};
                      slave_fermion::Bool=false)
    return -copy(adjoint(d_plus_d_min(elt, particle_symmetry, spin_symmetry; slave_fermion)))
end
const d⁻d⁺ = d_min_d_plus

@doc """
    u_min_d_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    u⁻d⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↑} e_{2,↓}`` that annihilates a spin-up particle at the first site and a spin-down particle at the second site.
The only nonzero matrix element corresponds to `|0,0⟩ <-- |↑,↓⟩`.
""" u_min_d_min
function u_min_d_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return u_min_d_min(ComplexF64, P, S; slave_fermion)
end
function u_min_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
                     slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h), I(h), dual(I(b)), dual(I(b)))][1, 1, 1, 2] = -sgn * 1
    return t
end
function u_min_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep};
                     slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h, 0), I(h, 0), dual(I(b, 1 // 2)), dual(I(b, -1 // 2)))] .= -sgn * 1
    return t
end
function u_min_d_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{<:Sector};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`u_min_d_min` is not symmetric under `U1Irrep` particle symmetry"))
end
function u_min_d_min(elt::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`u_min_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_min_d_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`u_min_d_min` is not symmetric under `U1Irrep` particle symmetry or under `SU2Irrep` spin symmetry"))
end
const u⁻d⁻ = u_min_d_min

@doc """
    d_min_u_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    d⁻u⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↓} e_{2,↑}`` that annihilates a spin-down particle at the first site and a spin-up particle at the second site.
The only nonzero matrix element corresponds to `|0,0⟩ <-- |↓,↑⟩`.
""" d_min_u_min
function d_min_u_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return d_min_u_min(ComplexF64, P, S; slave_fermion)
end
function d_min_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
                     slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h), I(h), dual(I(b)), dual(I(b)))][1, 1, 2, 1] = -sgn * 1
    return t
end
function d_min_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep};
                     slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h, 0), I(h, 0), dual(I(b, -1 // 2)), dual(I(b, 1 // 2)))] .= -sgn * 1
    return t
end
function d_min_u_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{<:Sector};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`d_min_u_min` is not symmetric under `U1Irrep` particle symmetry"))
end
function d_min_u_min(elt::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`d_min_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_min_u_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`d_min_u_min` is not symmetric under `U1Irrep` particle symmetry or under `SU2Irrep` particle symmetry"))
end
const d⁻u⁻ = d_min_u_min

@doc """
    u_min_u_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    u⁻u⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↑} e_{2,↑}`` that annihilates a spin-up particle at both sites.
The only nonzero matrix element corresponds to `|0,0⟩ <-- |↑,↑⟩`.
""" u_min_u_min
function u_min_u_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return u_min_u_min(ComplexF64, P, S; slave_fermion)
end
function u_min_u_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
                     slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h), I(h), dual(I(b)), dual(I(b)))][1, 1, 1, 1] = -sgn * 1
    return t
end
function u_min_u_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{<:Sector};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`u_min_u_min` is not symmetric under `U1Irrep` particle symmetry"))
end
function u_min_u_min(elt::Type{<:Number}, ::Type{<:Sector}, ::Type{U1Irrep};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`u_min_u_min` is not symmetric under `U1Irrep` spin symmetry"))
end
function u_min_u_min(elt::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`u_min_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_min_u_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`u_min_u_min` is not symmetric under `U1Irrep` particle symmetry or under `U1Irrep` particle symmetry"))
end
function u_min_u_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`u_min_u_min` is not symmetric under `U1Irrep` particle symmetry or under `SU2Irrep` particle symmetry"))
end
const u⁻u⁻ = u_min_u_min

@doc """
    u_plus_u_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    u⁺u⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↑} e†_{2,↑}`` that annihilates a spin-up particle at both sites.
""" u_plus_u_plus
function u_plus_u_plus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return u_plus_u_plus(ComplexF64, P, S; slave_fermion)
end
function u_plus_u_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                       spin_symmetry::Type{<:Sector};
                       slave_fermion::Bool=false)
    return -copy(adjoint(u_min_u_min(elt, particle_symmetry, spin_symmetry; slave_fermion)))
end
const u⁺u⁺ = u_plus_u_plus

@doc """
    d_min_d_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    d⁻d⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↓} e_{2,↓}`` that annihilates a spin-down particle at both sites.
The only nonzero matrix element corresponds to `|0,0⟩ <-- |↓,↓⟩`.
""" d_min_d_min
function d_min_d_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return d_min_d_min(ComplexF64, P, S; slave_fermion)
end
function d_min_d_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
                     slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h), I(h), dual(I(b)), dual(I(b)))][1, 1, 2, 2] = -sgn * 1
    return t
end
function d_min_d_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{<:Sector};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`d_min_d_min` is not symmetric under `U1Irrep` particle symmetry"))
end
function d_min_d_min(elt::Type{<:Number}, ::Type{<:Sector}, ::Type{U1Irrep};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`d_min_d_min` is not symmetric under `U1Irrep` spin symmetry"))
end
function d_min_d_min(elt::Type{<:Number}, ::Type{<:Sector}, ::Type{SU2Irrep};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`d_min_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_min_d_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`d_min_d_min` is not symmetric under `U1Irrep` particle symmetry or under `U1Irrep` particle symmetry"))
end
function d_min_d_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep};
                     slave_fermion::Bool=false)
    throw(ArgumentError("`d_min_d_min` is not symmetric under `U1Irrep` particle symmetry or under `SU2Irrep` particle symmetry"))
end
const d⁻d⁻ = d_min_d_min

@doc """
    d_plus_d_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    d⁺d⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↓} e†_{2,↓}`` that annihilates a spin-down particle at both sites.
""" d_plus_d_plus
function d_plus_d_plus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return d_plus_d_plus(ComplexF64, P, S; slave_fermion)
end
function d_plus_d_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                       spin_symmetry::Type{<:Sector};
                       slave_fermion::Bool=false)
    return -copy(adjoint(d_min_d_min(elt, particle_symmetry, spin_symmetry; slave_fermion)))
end
const d⁺d⁺ = d_plus_d_plus

@doc """
    e_plus_e_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    e⁺e⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
This is the sum of `u_plus_u_min` and `d_plus_d_min`.
""" e_plus_e_min
function e_plus_e_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_plus_e_min(ComplexF64, P, S; slave_fermion)
end
function e_plus_e_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                      spin_symmetry::Type{<:Sector};
                      slave_fermion::Bool=false)
    return u_plus_u_min(elt, particle_symmetry, spin_symmetry; slave_fermion) +
           d_plus_d_min(elt, particle_symmetry, spin_symmetry; slave_fermion)
end
function e_plus_e_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, SU2Irrep; slave_fermion)
    I = sectortype(t)
    (h, b) = slave_fermion ? (1, 0) : (0, 1)
    f1 = only(fusiontrees((I(h, 0), I(b, 1 // 2)), I(1, 1 // 2)))
    f2 = only(fusiontrees((I(b, 1 // 2), I(h, 0)), I(1, 1 // 2)))
    t[f1, f2][1, 1, 1, 1] = 1
    return t
end
function e_plus_e_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, U1Irrep, SU2Irrep; slave_fermion)
    I = sectortype(t)
    (h, b) = slave_fermion ? (1, 0) : (0, 1)
    f1 = only(fusiontrees((I(h, 0, 0), I(b, 1, 1 // 2)), I(1, 1, 1 // 2)))
    f2 = only(fusiontrees((I(b, 1, 1 // 2), I(h, 0, 0)), I(1, 1, 1 // 2)))
    t[f1, f2][1, 1, 1, 1] = 1
    return t
end

const e⁺e⁻ = e_plus_e_min

@doc """
    e_min_e_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    e⁻e⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
This is the sum of `u_min_u_plus` and `d_min_d_plus`.
""" e_min_e_plus
function e_min_e_plus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_min_e_plus(ComplexF64, P, S; slave_fermion)
end
function e_min_e_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                      spin_symmetry::Type{<:Sector};
                      slave_fermion::Bool=false)
    return -copy(adjoint(e_plus_e_min(elt, particle_symmetry, spin_symmetry; slave_fermion)))
end
const e⁻e⁺ = e_min_e_plus

@doc """
    singlet_min(elt, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    singlet⁻(elt, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body singlet operator ``(e_{1,↓} e_{2,↑} - e_{1,↓} e_{2,↑}) / sqrt(2)``.
""" singlet_min
function singlet_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return singlet_min(ComplexF64, P, S; slave_fermion)
end
function singlet_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                     spin_symmetry::Type{<:Sector};
                     slave_fermion::Bool=false)
    return (u_min_d_min(elt, particle_symmetry, spin_symmetry; slave_fermion) -
            d_min_u_min(elt, particle_symmetry, spin_symmetry; slave_fermion)) / sqrt(2)
end
const singlet⁻ = singlet_min

@doc """
    e_hopping([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)
    e_hop([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}], [spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the two-body operator that describes a particle that hops between the first and the second site.
""" e_hopping
function e_hopping(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_hopping(ComplexF64, P, S; slave_fermion)
end
function e_hopping(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                   spin_symmetry::Type{<:Sector};
                   slave_fermion::Bool=false)
    return e_plus_e_min(elt, particle_symmetry, spin_symmetry; slave_fermion) -
           e_min_e_plus(elt, particle_symmetry, spin_symmetry; slave_fermion)
end
const e_hop = e_hopping

@doc """
    S_plus_S_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    S⁺S⁻(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator S⁺S⁻.
The only nonzero matrix element corresponds to `|↑,↓⟩ <-- |↓,↑⟩`.
""" S_plus_S_min
function S_plus_S_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_plus_S_min(ComplexF64, P, S; slave_fermion)
end
function S_plus_S_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), I(b), dual(I(b)), dual(I(b)))][1, 2, 2, 1] = 1
    return t
end
function S_plus_S_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1 // 2), I(b, -1 // 2), dual(I(b, -1 // 2)), dual(I(b, 1 // 2)))] .= 1
    return t
end
function S_plus_S_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), I(b, 1), dual(I(b, 1)), dual(I(b, 1)))][1, 2, 2, 1] = 1
    return t
end
function S_plus_S_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep};
                      slave_fermion::Bool=false)
    t = two_site_operator(elt, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1, 1 // 2), I(b, 1, -1 // 2), dual(I(b, 1, -1 // 2)), dual(I(b, 1, 1 // 2)))] .= 1
    return t
end
const S⁺S⁻ = S_plus_S_min

@doc """
    S_min_S_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    S⁻S⁺(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator S⁻S⁺.
The only nonzero matrix element corresponds to `|↓,↑⟩ <-- |↑,↓⟩`.
""" S_min_S_plus
function S_min_S_plus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_min_S_plus(ComplexF64, P, S; slave_fermion)
end
function S_min_S_plus(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                      spin_symmetry::Type{<:Sector};
                      slave_fermion::Bool=false)
    return copy(adjoint(S_plus_S_min(elt, particle_symmetry, spin_symmetry; slave_fermion)))
end
const S⁻S⁺ = S_min_S_plus

@doc """
    S_exchange(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the spin exchange operator S⋅S.
""" S_exchange
function S_exchange(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_exchange(ComplexF64, P, S; slave_fermion)
end
function S_exchange(elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
                    spin_symmetry::Type{<:Sector};
                    slave_fermion::Bool=false)
    Sz = S_z(elt, particle_symmetry, spin_symmetry; slave_fermion)
    return (1 / 2) * (S_plus_S_min(elt, particle_symmetry, spin_symmetry; slave_fermion)
                      +
                      S_min_S_plus(elt, particle_symmetry, spin_symmetry; slave_fermion)) +
           Sz ⊗ Sz
end
function S_exchange(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep};
                    slave_fermion::Bool=false)
    t = two_site_operator(elt, Trivial, SU2Irrep; slave_fermion)
    for (s, f) in fusiontrees(t)
        l3 = f.uncoupled[1][2].j
        l4 = f.uncoupled[2][2].j
        k = f.coupled[2].j
        t[s, f] .= (k * (k + 1) - l3 * (l3 + 1) - l4 * (l4 + 1)) / 2
    end
    return t
end
function S_exchange(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep};
                    slave_fermion::Bool=false)
    t = two_site_operator(elt, U1Irrep, SU2Irrep; slave_fermion)
    for (s, f) in fusiontrees(t)
        l3 = f.uncoupled[1][3].j
        l4 = f.uncoupled[2][3].j
        k = f.coupled[3].j
        t[s, f] .= (k * (k + 1) - l3 * (l3 + 1) - l4 * (l4 + 1)) / 2
    end
    return t
end

end
