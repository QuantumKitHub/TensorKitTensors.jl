#= Operators that act on t-J-type models
i.e. the local hilbert space consists of 
=#
module TJOperators

using TensorKit

export tj_space
export c_num, u_num, d_num, c_num_hole
export S_x, S_y, S_z, S_plus, S_min
export u_plus_u_min, d_plus_d_min
export u_min_u_plus, d_min_d_plus
export u_min_d_min, d_min_u_min
export c_plus_c_min, c_min_c_plus, c_singlet
export S_plusmin, S_minplus, S_exchange

export nꜛ, nꜜ, nʰ, n
export Sˣ, Sʸ, Sᶻ, S⁺, S⁻
export u⁺u⁻, d⁺d⁻, u⁻u⁺, d⁻d⁺, u⁻d⁻, d⁻u⁻
export c⁺c⁻, c⁻c⁺
export S⁻⁺, S⁺⁻

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
function single_site_operator(T, particle_symmetry::Type{<:Sector},
                              spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    V = tj_space(particle_symmetry, spin_symmetry; slave_fermion)
    return zeros(T, V ← V)
end

@doc """
    u_num(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the one-body operator that counts the number of spin-up electrons.
""" u_num
function u_num(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return u_num(ComplexF64, P, S; slave_fermion)
end
function u_num(T::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
               slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][1, 1] = 1
    return t
end
function u_num(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1 // 2), dual(I(b, 1 // 2)))][1, 1] = 1
    return t
end
function u_num(T, ::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`u_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_num(T, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][1, 1] = 1
    return t
end
function u_num(T, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1, 1 // 2), dual(I(b, 1, 1 // 2)))] .= 1
    return t
end
function u_num(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`u_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
const nꜛ = u_num

@doc """
    d_num(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body operator that counts the number of spin-down electrons.
""" d_num
function d_num(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return d_num(ComplexF64, P, S; slave_fermion)
end
function d_num(T::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
               slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][2, 2] = 1
    return t
end
function d_num(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, -1 // 2), dual(I(b, -1 // 2)))][1, 1] = 1
    return t
end
function d_num(T, ::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`d_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_num(T, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][2, 2] = 1
    return t
end
function d_num(T, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1, -1 // 2), dual(I(b, 1, -1 // 2)))] .= 1
    return t
end
function d_num(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`d_num` is not symmetric under `SU2Irrep` spin symmetry"))
end
const nꜜ = d_num

@doc """
    c_num(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body operator that counts the number of particles.
""" c_num
function c_num(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return c_num(ComplexF64, P, S; slave_fermion)
end
function c_num(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
               slave_fermion::Bool=false)
    return u_num(T, particle_symmetry, spin_symmetry; slave_fermion) +
           d_num(T, particle_symmetry, spin_symmetry; slave_fermion)
end
function c_num(T, ::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, SU2Irrep; slave_fermion)
    I = sectortype(t)
    if slave_fermion
        block(t, I(0, 1 // 2))[1, 1] = 1
        # block(t, I(0, 1 // 2))[2, 2] = 1
    else
        block(t, I(1, 1 // 2))[1, 1] = 1
        # block(t, I(1, 1 // 2))[2, 2] = 1
    end
    return t
end
function c_num(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, SU2Irrep; slave_fermion)
    I = sectortype(t)
    if slave_fermion
        block(t, I(0, 1, 1 // 2))[1, 1] = 1
        # block(t, I(0, 1, 1 // 2))[2, 2] = 1
    else
        block(t, I(1, 1, 1 // 2))[1, 1] = 1
        # block(t, I(1, 1, 1 // 2))[2, 2] = 1
    end
    return t
end
const n = c_num

@doc """
    c_num_hole(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body operator that counts the number of holes.
""" c_num_hole
function c_num_hole(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return c_num_hole(ComplexF64, P, S; slave_fermion)
end
function c_num_hole(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                    slave_fermion::Bool=false)
    iden = TensorKit.id(tj_space(particle_symmetry, spin_symmetry; slave_fermion))
    return iden - c_num(T, particle_symmetry, spin_symmetry; slave_fermion)
end
const nʰ = c_num_hole

@doc """
    S_x(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body spin-1/2 x-operator on the electrons.
""" S_x
function S_x(P::Type{<:Sector}=Trivial, S::Type{<:Sector}=Trivial;
             slave_fermion::Bool=false)
    return S_x(ComplexF64, P, S; slave_fermion)
end
function S_x(T::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][1, 2] = 0.5
    t[(I(b), dual(I(b)))][2, 1] = 0.5
    return t
end
function S_x(T::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][1, 2] = 0.5
    t[(I(b, 1), dual(I(b, 1)))][2, 1] = 0.5
    return t
end
const Sˣ = S_x

@doc """
    S_y(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body spin-1/2 x-operator on the electrons (only defined for `Trivial` symmetry). 
""" S_y
function S_y(P::Type{<:Sector}=Trivial, S::Type{<:Sector}=Trivial;
             slave_fermion::Bool=false)
    return S_y(ComplexF64, P, S; slave_fermion)
end
function S_y(T::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][1, 2] = -0.5im
    t[(I(b), dual(I(b)))][2, 1] = 0.5im
    return t
end
function S_y(T::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][1, 2] = -0.5im
    t[(I(b, 1), dual(I(b, 1)))][2, 1] = 0.5im
    return t
end
const Sʸ = S_y

@doc """
    S_z(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body spin-1/2 z-operator on the electrons. 
""" S_z
function S_z(P::Type{<:Sector}=Trivial, S::Type{<:Sector}=Trivial;
             slave_fermion::Bool=false)
    return S_z(ComplexF64, P, S; slave_fermion)
end
function S_z(T::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][1, 1] = 0.5
    t[(I(b), dual(I(b)))][2, 2] = -0.5
    return t
end
function S_z(T::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1 // 2), dual(I(b, 1 // 2)))] .= 0.5
    t[(I(b, -1 // 2), dual(I(b, -1 // 2)))] .= -0.5
    return t
end
function S_z(T::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][1, 1] = 0.5
    t[(I(b, 1), dual(I(b, 1)))][2, 2] = -0.5
    return t
end
function S_z(T::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1, 1 // 2), dual(I(b, 1, 1 // 2)))] .= 0.5
    t[(I(b, 1, -1 // 2), dual(I(b, 1, -1 // 2)))] .= -0.5
    return t
end
const Sᶻ = S_z

@doc """
    S_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the spin-plus operator.
""" S_plus
function S_plus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_plus(ComplexF64, P, S; slave_fermion)
end
function S_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                slave_fermion::Bool=false)
    return S_x(T, particle_symmetry, spin_symmetry; slave_fermion) +
           1im * S_y(T, particle_symmetry, spin_symmetry; slave_fermion)
end
const S⁺ = S_plus

@doc """
    S_min(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the spin-minus operator.
""" S_min
function S_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_min(ComplexF64, P, S; slave_fermion)
end
function S_min(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
               slave_fermion::Bool=false)
    return S_x(T, particle_symmetry, spin_symmetry; slave_fermion) -
           1im * S_y(T, particle_symmetry, spin_symmetry; slave_fermion)
end
const S⁻ = S_min

# Two site operators
# ------------------
function two_site_operator(T, particle_symmetry::Type{<:Sector},
                           spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    V = tj_space(particle_symmetry, spin_symmetry; slave_fermion)
    return zeros(T, V ⊗ V ← V ⊗ V)
end

@doc """
    u_plus_u_min(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↑}, e_{2,↑}`` that creates a spin-up electron at the first site and annihilates a spin-up electron at the second.
The only nonzero matrix element corresponds to `|↑0⟩ <-- |0↑⟩`.
""" u_plus_u_min
function u_plus_u_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return u_plus_u_min(ComplexF64, P, S; slave_fermion)
end
function u_plus_u_min(T, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, Trivial; slave_fermion)
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
function u_plus_u_min(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1 // 2), I(h, 0), dual(I(h, 0)), dual(I(b, 1 // 2)))] .= sgn * 1
    return t
end
function u_plus_u_min(T, ::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`u_plus_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_plus_u_min(T, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1), I(h, 0), dual(I(h, 0)), dual(I(b, 1)))][1, 1, 1, 1] = sgn * 1
    return t
end
function u_plus_u_min(T, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1, 1 // 2), I(h, 0, 0), dual(I(h, 0, 0)), dual(I(b, 1, 1 // 2)))] .= sgn * 1
    return t
end
function u_plus_u_min(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`u_plus_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
const u⁺u⁻ = u_plus_u_min

@doc """
    d_plus_d_min(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↓}, e_{2,↓}`` that creates a spin-down electron at the first site and annihilates a spin-down electron at the second.
The only nonzero matrix element corresponds to `|↓0⟩ <-- |0↓⟩`.
""" d_plus_d_min
function d_plus_d_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return d_plus_d_min(ComplexF64, P, S; slave_fermion)
end
function d_plus_d_min(T, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b), I(h), dual(I(h)), dual(I(b)))][2, 1, 1, 2] = sgn * 1
    return t
end
function d_plus_d_min(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, -1 // 2), I(h, 0), dual(I(h, 0)), dual(I(b, -1 // 2)))] .= sgn * 1
    return t
end
function d_plus_d_min(T, ::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`d_plus_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_plus_d_min(T, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1), I(h, 0), dual(I(h, 0)), dual(I(b, 1)))][2, 1, 1, 2] = sgn * 1
    return t
end
function d_plus_d_min(T, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1, -1 // 2), I(h, 0, 0), dual(I(h, 0, 0)), dual(I(b, 1, -1 // 2)))] .= sgn * 1
    return t
end
function d_plus_d_min(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`d_plus_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
const d⁺d⁻ = d_plus_d_min

@doc """
    u_min_u_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the Hermitian conjugate of `u_plus_u_min`, i.e.
``(e†_{1,↑}, e_{2,↑})† = -e_{1,↑}, e†_{2,↑}`` (note the extra minus sign). 
It annihilates a spin-up electron at the first site and creates a spin-up electron at the second.
The only nonzero matrix element corresponds to `|0↑⟩ <-- |↑0⟩`.
""" u_min_u_plus
function u_min_u_plus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return u_min_u_plus(ComplexF64, P, S; slave_fermion)
end
function u_min_u_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                      slave_fermion::Bool=false)
    return copy(adjoint(u_plus_u_min(T, particle_symmetry, spin_symmetry; slave_fermion)))
end
const u⁻u⁺ = u_min_u_plus

@doc """
    d_min_d_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the Hermitian conjugate of `d_plus_d_min`, i.e.
``(e†_{1,↓}, e_{2,↓})† = -e_{1,↓}, e†_{2,↓}`` (note the extra minus sign). 
It annihilates a spin-down electron at the first site and creates a spin-down electron at the second.
The only nonzero matrix element corresponds to `|0↓⟩ <-- |↓0⟩`.
""" d_min_d_plus
function d_min_d_plus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return d_min_d_plus(ComplexF64, P, S; slave_fermion)
end
function d_min_d_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                      slave_fermion::Bool=false)
    return copy(adjoint(d_plus_d_min(T, particle_symmetry, spin_symmetry; slave_fermion)))
end
const d⁻d⁺ = d_min_d_plus

@doc """
    u_min_d_min(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↑} e_{2,↓}`` that annihilates a spin-up particle at the first site and a spin-down particle at the second site.
The only nonzero matrix element corresponds to `|00⟩ <-- |↑↓⟩`.
""" u_min_d_min
function u_min_d_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return u_min_d_min(ComplexF64, P, S; slave_fermion)
end
function u_min_d_min(T, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h), I(h), dual(I(b)), dual(I(b)))][1, 1, 1, 2] = -sgn * 1
    return t
end
function u_min_d_min(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h, 0), I(h, 0), dual(I(b, 1 // 2)), dual(I(b, -1 // 2)))] .= -sgn * 1
    return t
end
function u_min_d_min(T, ::Type{U1Irrep}, ::Type{<:Sector}; slave_fermion::Bool=false)
    throw(ArgumentError("`u_min_d_min` is not symmetric under `U1Irrep` particle symmetry"))
end
function u_min_d_min(T, ::Type{<:Sector}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`u_min_d_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function u_min_d_min(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`u_min_d_min` is not symmetric under `U1Irrep` particle symmetry or under `SU2Irrep` spin symmetry"))
end
const u⁻d⁻ = u_min_d_min

@doc """
    d_min_u_min(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↓} e_{2,↑}`` that annihilates a spin-down particle at the first site and a spin-up particle at the second site.
The only nonzero matrix element corresponds to `|00⟩ <-- |↓↑⟩`.
""" d_min_u_min
function d_min_u_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return d_min_u_min(ComplexF64, P, S; slave_fermion)
end
function d_min_u_min(T, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h), I(h), dual(I(b)), dual(I(b)))][1, 1, 2, 1] = -sgn * 1
    return t
end
function d_min_u_min(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h, 0), I(h, 0), dual(I(b, -1 // 2)), dual(I(b, 1 // 2)))] .= -sgn * 1
    return t
end
function d_min_u_min(T, ::Type{U1Irrep}, ::Type{<:Sector}; slave_fermion::Bool=false)
    throw(ArgumentError("`d_min_u_min` is not symmetric under `U1Irrep` particle symmetry"))
end
function d_min_u_min(T, ::Type{<:Sector}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`d_min_u_min` is not symmetric under `SU2Irrep` spin symmetry"))
end
function d_min_u_min(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`d_min_u_min` is not symmetric under `U1Irrep` particle symmetry or under `SU2Irrep` particle symmetry"))
end
const d⁻u⁻ = d_min_u_min

@doc """
    c_plus_c_min(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
This is the sum of `u_plus_u_min` and `d_plus_d_min`.
""" c_plus_c_min
function c_plus_c_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return c_plus_c_min(ComplexF64, P, S; slave_fermion)
end
function c_plus_c_min(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                      slave_fermion::Bool=false)
    return u_plus_u_min(T, particle_symmetry, spin_symmetry; slave_fermion) +
           d_plus_d_min(T, particle_symmetry, spin_symmetry; slave_fermion)
end
function c_plus_c_min(T, ::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, SU2Irrep; slave_fermion)
    I = sectortype(t)
    if slave_fermion
        f1 = only(fusiontrees((I(1, 0), I(0, 1 // 2)), I(1, 1 // 2)))
        f2 = only(fusiontrees((I(0, 1 // 2), I(1, 0)), I(1, 1 // 2)))
        t[f1, f2][1, 1, 1, 1] = 1
        # t[f1, f2][1, 2, 2, 1] = 1
        # f3 = only(fusiontrees((I(0, 1 // 2), I(1, 0)), I(1, 1 // 2)))
        # f4 = only(fusiontrees((I(1, 0), I(0, 1 // 2)), I(1, 1 // 2)))
        # t[f3, f4][1, 1, 1, 1] = -1
        # t[f3, f4][2, 1, 1, 2] = -1
    else
        f1 = only(fusiontrees((I(0, 0), I(1, 1 // 2)), I(1, 1 // 2)))
        f2 = only(fusiontrees((I(1, 1 // 2), I(0, 0)), I(1, 1 // 2)))
        t[f1, f2][1, 1, 1, 1] = 1
        # t[f1, f2][1, 2, 2, 1] = 1
        # f3 = only(fusiontrees((I(1, 1 // 2), I(0, 0)), I(1, 1 // 2)))
        # f4 = only(fusiontrees((I(0, 0), I(1, 1 // 2)), I(1, 1 // 2)))
        # t[f3, f4][1, 1, 1, 1] = -1
        # t[f3, f4][2, 1, 1, 2] = -1
    end
    return t
end
function c_plus_c_min(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, SU2Irrep; slave_fermion)
    I = sectortype(t)
    if slave_fermion
        f1 = only(fusiontrees((I(1, 0, 0), I(0, 1, 1 // 2)), I(1, 1, 1 // 2)))
        f2 = only(fusiontrees((I(0, 1, 1 // 2), I(1, 0, 0)), I(1, 1, 1 // 2)))
        t[f1, f2][1, 1, 1, 1] = 1
        # t[f1, f2][1, 2, 2, 1] = 1
        # f3 = only(fusiontrees((I(0, 1, 1 // 2), I(1, 0, 0)), I(1, 1, 1 // 2)))
        # f4 = only(fusiontrees((I(1, 0, 0), I(0, 1, 1 // 2)), I(1, 1, 1 // 2)))
        # t[f3, f4][1, 1, 1, 1] = -1
        # t[f3, f4][2, 1, 1, 2] = -1
    else
        f1 = only(fusiontrees((I(0, 0, 0), I(1, 1, 1 // 2)), I(1, 1, 1 // 2)))
        f2 = only(fusiontrees((I(1, 1, 1 // 2), I(0, 0, 0)), I(1, 1, 1 // 2)))
        t[f1, f2][1, 1, 1, 1] = 1
        # t[f1, f2][1, 2, 2, 1] = 1
        # f3 = only(fusiontrees((I(1, 1, 1 // 2), I(0, 0, 0)), I(1, 1, 1 // 2)))
        # f4 = only(fusiontrees((I(0, 0, 0), I(1, 1, 1 // 2)), I(1, 1, 1 // 2)))
        # t[f3, f4][1, 1, 1, 1] = -1
        # t[f3, f4][2, 1, 1, 2] = -1
    end
    return t
end

const c⁺c⁻ = c_plus_c_min

@doc """
    c_min_c_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
This is the sum of `u_min_u_plus` and `d_min_d_plus`.
""" c_min_c_plus
function c_min_c_plus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return c_min_c_plus(ComplexF64, P, S; slave_fermion)
end
function c_min_c_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                      slave_fermion::Bool=false)
    return copy(adjoint(c_plus_c_min(T, particle_symmetry, spin_symmetry; slave_fermion)))
end
const c⁻c⁺ = c_min_c_plus

@doc """
    c_singlet(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body singlet operator ``(e_{1,↓} e_{2,↑} - e_{1,↓} e_{2,↑}) / sqrt(2)``.
""" c_singlet
function c_singlet(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return c_singlet(ComplexF64, P, S; slave_fermion)
end
function c_singlet(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                   slave_fermion::Bool=false)
    return (u_min_d_min(T, particle_symmetry, spin_symmetry; slave_fermion) -
            d_min_u_min(T, particle_symmetry, spin_symmetry; slave_fermion)) / sqrt(2)
end

@doc """
    S_plusmin(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator S⁺S⁻.
The only nonzero matrix element corresponds to `|↑↓⟩ <-- |↓↑⟩`.
""" S_plusmin
function S_plusmin(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_plusmin(ComplexF64, P, S; slave_fermion)
end
function S_plusmin(T, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), I(b), dual(I(b)), dual(I(b)))][1, 2, 2, 1] = 1
    return t
end
function S_plusmin(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1 // 2), I(b, -1 // 2), dual(I(b, -1 // 2)), dual(I(b, 1 // 2)))] .= 1
    return t
end
function S_plusmin(T, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), I(b, 1), dual(I(b, 1)), dual(I(b, 1)))][1, 2, 2, 1] = 1
    return t
end
function S_plusmin(T, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1, 1 // 2), I(b, 1, -1 // 2), dual(I(b, 1, -1 // 2)), dual(I(b, 1, 1 // 2)))] .= 1
    return t
end
const S⁺⁻ = S_plusmin

@doc """
    S_minplus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator S⁻S⁺.
The only nonzero matrix element corresponds to `|↓↑⟩ <-- |↑↓⟩`.
""" S_minplus
function S_minplus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_minplus(ComplexF64, P, S; slave_fermion)
end
function S_minplus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                   slave_fermion::Bool=false)
    return copy(adjoint(S_plusmin(T, particle_symmetry, spin_symmetry; slave_fermion)))
end
const S⁻⁺ = S_minplus

@doc """
    S_exchange(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the spin exchange operator S⋅S.
""" S_exchange
function S_exchange(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_exchange(ComplexF64, P, S; slave_fermion)
end
function S_exchange(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                    slave_fermion::Bool=false)
    Sz = S_z(T, particle_symmetry, spin_symmetry; slave_fermion)
    return (1 / 2) * (S_plusmin(T, particle_symmetry, spin_symmetry; slave_fermion)
                      +
                      S_minplus(T, particle_symmetry, spin_symmetry; slave_fermion)) +
           Sz ⊗ Sz
end
function S_exchange(T, ::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, SU2Irrep; slave_fermion)

    for (s, f) in fusiontrees(t)
        l3 = f.uncoupled[1][2].j
        l4 = f.uncoupled[2][2].j
        k = f.coupled[2].j
        t[s, f] .= (k * (k + 1) - l3 * (l3 + 1) - l4 * (l4 + 1)) / 2
    end
    return t
end
function S_exchange(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, SU2Irrep; slave_fermion)

    for (s, f) in fusiontrees(t)
        l3 = f.uncoupled[1][3].j
        l4 = f.uncoupled[2][3].j
        k = f.coupled[3].j
        t[s, f] .= (k * (k + 1) - l3 * (l3 + 1) - l4 * (l4 + 1)) / 2
    end
    return t
end

end
