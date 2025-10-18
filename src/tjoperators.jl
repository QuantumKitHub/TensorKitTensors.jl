module TJOperators

using LinearAlgebra
using TensorKit
using ..HubbardOperators: HubbardOperators, hubbard_space
import ..TensorKitTensors: _fuse_ids

export tj_space, tj_projector
export e_num, u_num, d_num, h_num
export S_x, S_y, S_z, S_plus, S_min
export u_plus_u_min, d_plus_d_min
export u_min_u_plus, d_min_d_plus
export u_min_d_min, d_min_u_min
export u_plus_d_plus, d_plus_u_plus
export u_min_u_min, d_min_d_min
export u_plus_u_plus, d_plus_d_plus
export e_plus_e_min, e_min_e_plus, e_hopping
export singlet_plus, singlet_min
export S_plus_S_min, S_min_S_plus, S_exchange

export nꜛ, nꜜ, nʰ, n
export Sˣ, Sʸ, Sᶻ, S⁺, S⁻
export u⁺u⁻, d⁺d⁻, u⁻u⁺, d⁻d⁺
export u⁻d⁻, d⁻u⁻, u⁺d⁺, d⁺u⁺
export u⁻u⁻, u⁺u⁺, d⁻d⁻, d⁺d⁺
export e⁺e⁻, e⁻e⁺, e_hop
export singlet⁺, singlet⁻
export S⁻S⁺, S⁺S⁻

@doc """
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
""" tj_space
tj_space(::Type{Trivial} = Trivial, ::Type{Trivial} = Trivial; slave_fermion::Bool = false) =
    Vect[FermionParity](0 ⊻ slave_fermion => 1, 1 ⊻ slave_fermion => 2)
tj_space(::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool = false) =
    Vect[FermionParity ⊠ U1Irrep](
    (0 ⊻ slave_fermion, 0) => 1, (1 ⊻ slave_fermion, 1 // 2) => 1, (1 ⊻ slave_fermion, -1 // 2) => 1
)
tj_space(::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool = false) =
    Vect[FermionParity ⊠ SU2Irrep]((0 ⊻ slave_fermion, 0) => 1, (1 ⊻ slave_fermion, 1 // 2) => 1)
tj_space(::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool = false) =
    Vect[FermionParity ⊠ U1Irrep]((0 ⊻ slave_fermion, 0) => 1, (1 ⊻ slave_fermion, 1) => 2)
tj_space(::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool = false) =
    Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep](
    (0 ⊻ slave_fermion, 0, 0) => 1, (1 ⊻ slave_fermion, 1, 1 // 2) => 1, (1 ⊻ slave_fermion, 1, -1 // 2) => 1
)
tj_space(::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool = false) =
    Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep](
    (0 ⊻ slave_fermion, 0, 0) => 1, (1 ⊻ slave_fermion, 1, 1 // 2) => 1
)

@doc """
    slave_fermion_auxiliary_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the auxiliary space to add a fermion-Z2 charge
to the t-J space and switch to the slave fermion basis.
""" slave_fermion_auxiliary_space
function slave_fermion_auxiliary_space(::Type{Trivial}, ::Type{Trivial})
    return Vect[FermionParity](1 => 1)
end
function slave_fermion_auxiliary_space(::Type{Trivial}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ U1Irrep]((1, 0) => 1)
end
function slave_fermion_auxiliary_space(::Type{Trivial}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ SU2Irrep]((1, 0) => 1)
end
function slave_fermion_auxiliary_space(::Type{U1Irrep}, ::Type{Trivial})
    return Vect[FermionParity ⊠ U1Irrep]((1, 0) => 1)
end
function slave_fermion_auxiliary_space(::Type{U1Irrep}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((1, 0, 0) => 1)
end
function slave_fermion_auxiliary_space(::Type{U1Irrep}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((1, 0, 0) => 1)
end

"""
    tj_projector(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Projection operator from Hubbard space to t-J space (under usual basis, i.e. `slave_fermion = false`).
The scalartype is `Int` to avoid floating point errors.
"""
function tj_projector(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    Vhub = hubbard_space(particle_symmetry, spin_symmetry)
    VtJ = tj_space(particle_symmetry, spin_symmetry)
    proj = zeros(Int, Vhub → VtJ)

    for (f1, f2) in fusiontrees(proj)
        proj[f1, f2][diagind(proj[f1, f2])] .= 1
    end
    return proj
end

for (opname, alias) in zip(
        (
            :e_num, :u_num, :d_num, :h_num,
            :S_x, :S_y, :S_z, :S_plus, :S_min,
            :u_plus_u_min, :d_plus_d_min, :u_min_u_plus, :d_min_d_plus,
            :u_min_d_min, :d_min_u_min, :u_plus_d_plus, :d_plus_u_plus,
            :u_min_u_min, :d_min_d_min, :u_plus_u_plus, :d_plus_d_plus,
            :e_plus_e_min, :e_min_e_plus, :e_hopping,
            :singlet_plus, :singlet_min,
            :S_plus_S_min, :S_min_S_plus, :S_exchange,
        ), (
            :n, :nꜛ, :nꜜ, :nʰ,
            :Sˣ, :Sʸ, :Sᶻ, :S⁺, :S⁻,
            :u⁺u⁻, :d⁺d⁻, :u⁻u⁺, :d⁻d⁺,
            :u⁻d⁻, :d⁻u⁻, :u⁺d⁺, :d⁺u⁺,
            :u⁻u⁻, :u⁺u⁺, :d⁻d⁻, :d⁺d⁺,
            :e⁺e⁻, :e⁻e⁺, :e_hop,
            :singlet⁺, :singlet⁻,
            :S⁻S⁺, :S⁺S⁻, nothing,
        )
    )
    # copy over the docstrings
    @eval begin
        @doc (@doc HubbardOperators.$opname) $opname
    end

    # apply projector on Hubbard operator
    @eval function $opname(
            elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
            slave_fermion::Bool = false
        )
        particle_symmetry == SU2Irrep &&
            throw(ArgumentError("t-J model does not have ``SU(2)`` particle symmetry."))
        op_H = HubbardOperators.$opname(elt, particle_symmetry, spin_symmetry)
        proj = tj_projector(particle_symmetry, spin_symmetry)
        N = numin(op_H)
        (N > 1) && (proj = reduce(⊗, ntuple(Returns(proj), N)))
        op_tJ = proj * op_H * proj'
        slave_fermion || return op_tJ

        Vaux = slave_fermion_auxiliary_space(particle_symmetry, spin_symmetry)
        return _fuse_ids(op_tJ, ntuple(Returns(Vaux), N))
    end

    # define alias
    isnothing(alias) || @eval begin
        const $alias = $opname
    end
end

end
