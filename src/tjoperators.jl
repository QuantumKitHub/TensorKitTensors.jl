module TJOperators

using LinearAlgebra
using TensorKit
import ..HubbardOperators as Hub
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
function tj_space(
        ::Type{Trivial} = Trivial, ::Type{Trivial} = Trivial; slave_fermion::Bool = false
    )
    return slave_fermion ? Vect[FermionParity](0 => 2, 1 => 1) :
        Vect[FermionParity](0 => 1, 1 => 2)
end
function tj_space(::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool = false)
    return if slave_fermion
        Vect[FermionParity ⊠ U1Irrep]((1, 0) => 1, (0, 1 // 2) => 1, (0, -1 // 2) => 1)
    else
        Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
    end
end
function tj_space(::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool = false)
    return slave_fermion ? Vect[FermionParity ⊠ SU2Irrep]((1, 0) => 1, (0, 1 // 2) => 1) :
        Vect[FermionParity ⊠ SU2Irrep]((0, 0) => 1, (1, 1 // 2) => 1)
end
function tj_space(::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool = false)
    return if slave_fermion
        Vect[FermionParity ⊠ U1Irrep]((1, 0) => 1, (0, 1) => 2)
    else
        Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 2)
    end
end
function tj_space(::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool = false)
    return if slave_fermion
        Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep](
            (1, 0, 0) => 1, (0, 1, 1 // 2) => 1, (0, 1, -1 // 2) => 1
        )
    else
        Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep](
            (0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (1, 1, -1 // 2) => 1
        )
    end
end
function tj_space(::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool = false)
    return if slave_fermion
        Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((1, 0, 0) => 1, (0, 1, 1 // 2) => 1)
    else
        Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1)
    end
end

"""
    slave_fermion_auxiliary_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the auxiliary space to add a fermion-Z2 charge
to the t-J space and switch to the slave fermion basis.
"""
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

Projection operator from Hubbard space to t-J space (under usual basis, i.e. `slave_fermion = false`). The scalartype is `Int`.
"""
function tj_projector(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    Vhub = Hub.hubbard_space(particle_symmetry, spin_symmetry)
    VtJ = tj_space(particle_symmetry, spin_symmetry)
    proj = zeros(Int, Vhub → VtJ)
    for (f1, f2) in fusiontrees(proj)
        proj[f1, f2][diagind(proj[f1, f2])] .= 1
    end
    return proj
end

# Single-site operators
for opname in (
        :e_num, :u_num, :d_num,
        :S_x, :S_y, :S_z, :S_plus, :S_min,
        :n, :nꜛ, :nꜜ,
        :Sˣ, :Sʸ, :Sᶻ, :S⁺, :S⁻,
    )
    @eval begin
        function ($opname)(args...; slave_fermion::Bool = false)
            psymm, ssymm = args[end - 1], args[end]
            if psymm == SU2Irrep
                error("t-J model doesn't have SU(2) particle symmetry.")
            end
            opHub = Hub.$opname(args...)
            proj = tj_projector(psymm, ssymm)
            optJ = proj * opHub * proj'
            if slave_fermion
                Vaux = slave_fermion_auxiliary_space(psymm, ssymm)
                optJ = _fuse_ids(optJ, (Vaux,))
            end
            return optJ
        end
    end
end

@doc """
    h_num(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    nʰ(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body operator that counts the number of holes.
""" h_num
function h_num(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool = false)
    return h_num(ComplexF64, P, S; slave_fermion)
end
function h_num(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false
    )
    iden = TensorKit.id(elt, tj_space(particle_symmetry, spin_symmetry; slave_fermion))
    return iden - e_num(elt, particle_symmetry, spin_symmetry; slave_fermion)
end
const nʰ = h_num

# Two-site operators
for opname in (
        :u_plus_u_min, :d_plus_d_min,
        :u_min_u_plus, :d_min_d_plus,
        :u_min_d_min, :d_min_u_min,
        :u_plus_d_plus, :d_plus_u_plus,
        :u_min_u_min, :d_min_d_min,
        :u_plus_u_plus, :d_plus_d_plus,
        :e_plus_e_min, :e_min_e_plus, :e_hopping,
        :singlet_plus, :singlet_min,
        :S_plus_S_min, :S_min_S_plus, :S_exchange,
        :u⁺u⁻, :d⁺d⁻, :u⁻u⁺, :d⁻d⁺,
        :u⁻d⁻, :d⁻u⁻, :u⁺d⁺, :d⁺u⁺,
        :u⁻u⁻, :u⁺u⁺, :d⁻d⁻, :d⁺d⁺,
        :e⁺e⁻, :e⁻e⁺, :e_hop,
        :singlet⁺, :singlet⁻,
        :S⁻S⁺, :S⁺S⁻,
    )
    @eval begin
        function ($opname)(args...; slave_fermion::Bool = false)
            psymm, ssymm = args[end - 1], args[end]
            if psymm == SU2Irrep
                error("t-J model doesn't have SU(2) particle symmetry.")
            end
            opHub = Hub.$opname(args...)
            proj = tj_projector(psymm, ssymm)
            proj = proj ⊗ proj
            optJ = proj * opHub * proj'
            if slave_fermion
                Vaux = slave_fermion_auxiliary_space(psymm, ssymm)
                optJ = _fuse_ids(optJ, (Vaux, Vaux))
            end
            return optJ
        end
    end
end

end
