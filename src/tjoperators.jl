module TJOperators

using LinearAlgebra
using TensorKit
import ..HubbardOperators
import ..TensorKitTensors: fuse_local_operators

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

export transform_slave_fermion

@doc """
    tj_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the local hilbert space for a t-J-type model with the given particle and spin symmetries.
The basis consists of ``|0⟩``, ``|↑⟩ = u⁺|0⟩`` and ``|↓⟩ = d⁺|0⟩``.
When `slave_fermion` is `true`, the basis consists of ``h⁺|0⟩``, ``bꜛ⁺|0⟩`` and ``bꜜ⁺|0⟩``,
where ``h`` is the fermionic holon operator, and ``bꜛ``, ``bꜜ`` are bosonic spinon operators.

The possible symmetries are:
- Particle number : `Trivial`, `U1Irrep`
- Spin            : `Trivial`, `U1Irrep`, `SU2Irrep`.
""" tj_space
function tj_space(
        particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
        slave_fermion::Bool = false
    )
    V = _tj_space(particle_symmetry, spin_symmetry)
    if slave_fermion
        charge = slave_fermion_auxiliary_charge(sectortype(V))
        V_aux = spacetype(V)(charge => 1)
        V = fuse(V, V_aux)
    end
    return V
end

_tj_space(::Type{Trivial} = Trivial, ::Type{Trivial} = Trivial) =
    Vect[FermionParity](0 => 1, 1 => 2)
_tj_space(::Type{Trivial}, ::Type{U1Irrep}) =
    Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
_tj_space(::Type{Trivial}, ::Type{SU2Irrep}) =
    Vect[FermionParity ⊠ SU2Irrep]((0, 0) => 1, (1, 1 // 2) => 1)
_tj_space(::Type{U1Irrep}, ::Type{Trivial}) =
    Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 2)
_tj_space(::Type{U1Irrep}, ::Type{U1Irrep}) =
    Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (1, 1, -1 // 2) => 1)
_tj_space(::Type{U1Irrep}, ::Type{SU2Irrep}) =
    Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1)

"""
    tj_projector(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Projection operator from Hubbard space to t-J space.
The scalartype is `Int` to avoid floating point errors.
"""
function tj_projector(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    Vhub = HubbardOperators.hubbard_space(particle_symmetry, spin_symmetry)
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
        hub_doc = (@doc HubbardOperators.$opname)
        tJ_doc = if hub_doc isa Base.Docs.DocStr
            hub_doc.text[1]
        else
            # compatibility with Julia 1.10 (hub_doc is a Markdown.MD object)
            string(hub_doc)
        end
        tJ_doc = replace(tJ_doc, "[spin_symmetry::Type{<:Sector}])" => "[spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)") * "Use `slave_fermion = true` to switch to the slave-fermion basis.\n"
        @doc (tJ_doc) $opname
    end

    # default arguments
    @eval $opname(
        particle_symmetry::Type{<:Sector} = Trivial, spin_symmetry::Type{<:Sector} = Trivial;
        slave_fermion::Bool = false
    ) = $opname(ComplexF64, particle_symmetry, spin_symmetry; slave_fermion)
    @eval $opname(elt::Type{<:Number}; slave_fermion::Bool = false) =
        $opname(elt, Trivial, Trivial; slave_fermion)

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
        op = proj * op_H * proj'
        if slave_fermion
            op = transform_slave_fermion(op)
        end
        return op
    end

    # define alias
    isnothing(alias) || @eval begin
        const $alias = $opname
    end
end

slave_fermion_auxiliary_charge(::Type{FermionParity}) = FermionParity(1)
slave_fermion_auxiliary_charge(::Type{ProductSector{T}}) where {T} =
    mapreduce(⊠, fieldtypes(T)) do I
    I === FermionParity ? FermionParity(1) : one(I)
end

"""
    transform_slave_fermion(O::AbstractTensorMap)

Transform the given operator to the slave-fermion basis, which is related to the usual t-J basis by

| tJ basis | slave-fermion |
| -------- | ------------- |
|   |0⟩    |      h⁺|0⟩    |
|  u⁺|0⟩   |     bꜛ⁺|0⟩    |
|  d⁺|0⟩   |     bꜜ⁺|0⟩    |

where ``h`` is the fermionic holon operator, and ``bꜛ``, ``bꜜ`` are bosonic spinon operators.
"""
function transform_slave_fermion(O::AbstractTensorMap)
    (N = numin(O)) == numout(O) || throw(ArgumentError("not a valid operator"))
    aux_charge = slave_fermion_auxiliary_charge(sectortype(O))
    aux_space = spacetype(O)(aux_charge => 1)
    aux_operator = id(Int, aux_space^N)

    return fuse_local_operators(O, aux_operator)
end

end
