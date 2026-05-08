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
export singlet_plus_singlet_min
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

const _docs_basis_table = """
```
| label | tJ basis | slave-fermion |
| ----- | -------- | ------------- |
|  |0⟩  |   |∅⟩    |      h⁺|∅⟩    |
|  |↑⟩  |  u⁺|∅⟩   |     bꜛ⁺|∅⟩    |
|  |↓⟩  |  d⁺|∅⟩   |     bꜜ⁺|∅⟩    |
```
"""

@doc """
    tj_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the local hilbert space for a t-J-type model with the given particle and spin symmetries.
The basis consists of the following states:

$_docs_basis_table

- `|∅⟩` is the vacuum state;
- `u` and `d` denote fermionic spin-up and spin-down operators;
- in the slave-fermion representation, ``h`` is the fermionic holon operator, and ``bꜛ``, ``bꜜ`` are bosonic spinon operators.

The possible symmetries are:
- Particle number : `Trivial`, `U1Irrep`
- Spin            : `Trivial`, `U1Irrep`, `SU2Irrep`.
""" tj_space
function tj_space(
        particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
        slave_fermion::Bool = false
    )
    V = if particle_symmetry === Trivial
        if spin_symmetry === Trivial
            Vect[FermionParity](0 => 1, 1 => 2)
        elseif spin_symmetry === U1Irrep
            Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
        elseif spin_symmetry === SU2Irrep
            Vect[FermionParity ⊠ SU2Irrep]((0, 0) => 1, (1, 1 // 2) => 1)
        else
            throw(ArgumentError("Invalid symmetry"))
        end
    elseif particle_symmetry === U1Irrep
        if spin_symmetry === Trivial
            Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 2)
        elseif spin_symmetry === U1Irrep
            Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (1, 1, -1 // 2) => 1)
        elseif spin_symmetry === SU2Irrep
            Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1)
        else
            throw(ArgumentError("Invalid symmetry"))
        end
    else
        throw(ArgumentError("Invalid symmetry"))
    end

    return slave_fermion ? transform_slave_fermion(V) : V
end

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
        tJ_doc = if occursin("[spin_symmetry::Type{<:Sector}])", tJ_doc)
            replace(tJ_doc, "[spin_symmetry::Type{<:Sector}])" => "[spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)")
        else
            replace(tJ_doc, "spin_symmetry::Type{<:Sector})" => "spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)")
        end
        tJ_doc = tJ_doc * "Use `slave_fermion = true` to switch to the slave-fermion basis.\n"
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
        return slave_fermion ? transform_slave_fermion(op) : op
    end

    # define alias
    isnothing(alias) || @eval begin
        const $alias = $opname
    end
end

function three_site_operator(
        elt::Type{<:Number}, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false,
    )
    V = tj_space(particle_symmetry, spin_symmetry; slave_fermion)
    return zeros(elt, V^3 ← V^3)
end

@doc """
    singlet_plus_singlet_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Returns the 3-site term ``O_{ijk} = A^†_{ij} A_{jk}``, where
``A^†_{ij} = (e^†_{1,↑} e^†_{2,↓} - e^†_{1,↓} e^†_{2,↑}) / \\sqrt{2}``.
It describes the hopping of a singlet pair from bond `(j,k)` to bond `(i,j)`.
""" singlet_plus_singlet_min
function singlet_plus_singlet_min(
        P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool = false
    )
    return singlet_plus_singlet_min(ComplexF64, P, S; slave_fermion)
end
function singlet_plus_singlet_min(elt::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)
    #=
                -5      -6
            ┌---┴-------┴---┐
            |     A_{jk}    |
            └---┬-------┬---┘
        -4      1       -3
    ┌---┴-------┴---┐
    |    A†_{ij}    |
    └---┬-------┬---┘
        -1      -2
        i       j       k
    =#
    singp = singlet_plus(elt, particle_symmetry, spin_symmetry)
    singm = singp'
    @tensor t[-1 -2 -3; -4 -5 -6] := singp[-1 -2; -4 1] * singm[1 -3; -5 -6]
    return slave_fermion ? transform_slave_fermion(t) : t
end
#=
The 3-site term can be expanded as
```
    O_{ijk} = ∑_σ (c†_{iσ} c†_{jσ̄} c_{jσ̄} c_{kσ} - c†_{iσ} c†_{jσ̄} c_{jσ} c_{kσ̄})
```
The only nonzero elements are given by
```
    + c†_{iσ} c†_{jσ̄} c_{jσ̄} c_{kσ} |0σ̄σ⟩ = - |σσ̄0⟩
    - c†_{iσ} c†_{jσ̄} c_{jσ} c_{kσ̄} |0σσ̄⟩ = + |σσ̄0⟩
```
leading to
```
    |0,↓,↑⟩ -> -|↑,↓,0⟩,    |0,↑,↓⟩ -> -|↓,↑,0⟩
    |0,↓,↑⟩ -> |↓,↑,0⟩,     |0,↑,↓⟩ -> |↑,↓,0⟩
```
=#
function singlet_plus_singlet_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool = false)
    t = three_site_operator(elt, U1Irrep, Trivial)
    S = sectortype(t)
    spin, hole = S(1, 1), S(0, 0)
    idx = (spin, spin, hole, dual(hole), dual(spin), dual(spin))
    t[idx][1, 2, 1, 1, 2, 1] = -1
    t[idx][2, 1, 1, 1, 1, 2] = -1
    t[idx][2, 1, 1, 1, 2, 1] = 1
    t[idx][1, 2, 1, 1, 1, 2] = 1
    return slave_fermion ? transform_slave_fermion(t) : t
end
function singlet_plus_singlet_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool = false)
    t = three_site_operator(elt, U1Irrep, U1Irrep)
    S = sectortype(t)
    u, d, h = S(1, 1, 1 // 2), S(1, 1, -1 // 2), S(0, 0, 0)
    t[(u, d, h, dual(h), dual(d), dual(u))] .= -1
    t[(d, u, h, dual(h), dual(u), dual(d))] .= -1
    t[(u, d, h, dual(h), dual(u), dual(d))] .= 1
    t[(d, u, h, dual(h), dual(d), dual(u))] .= 1
    return slave_fermion ? transform_slave_fermion(t) : t
end
function singlet_plus_singlet_min(elt::Type{<:Number}, ::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool = false)
    t = three_site_operator(elt, Trivial, SU2Irrep)
    S = sectortype(t)
    f1 = only(fusiontrees((S(1, 1 // 2), S(1, 1 // 2), S(0, 0)), S(0, 0)))
    f2 = only(fusiontrees((S(0, 0), S(1, 1 // 2), S(1, 1 // 2)), S(0, 0)))
    t[f1, f2] .= 1
    return slave_fermion ? transform_slave_fermion(t) : t
end
function singlet_plus_singlet_min(elt::Type{<:Number}, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool = false)
    t = three_site_operator(elt, U1Irrep, SU2Irrep)
    S = sectortype(t)
    f1 = only(fusiontrees((S(1, 1, 1 // 2), S(1, 1, 1 // 2), S(0, 0, 0)), S(0, 2, 0)))
    f2 = only(fusiontrees((S(0, 0, 0), S(1, 1, 1 // 2), S(1, 1, 1 // 2)), S(0, 2, 0)))
    t[f1, f2] .= 1
    return slave_fermion ? transform_slave_fermion(t) : t
end

slave_fermion_auxiliary_charge(::Type{FermionParity}) = FermionParity(1)
slave_fermion_auxiliary_charge(::Type{ProductSector{T}}) where {T} =
    mapreduce(⊠, fieldtypes(T)) do I
    I === FermionParity ? FermionParity(1) : one(I)
end

@doc """
    transform_slave_fermion(O::AbstractTensorMap)
    transform_slave_fermion(V::ElementarySpace)

Transform the given operator to the slave-fermion basis, which is related to the usual t-J basis by

$_docs_basis_table

where ``h`` is the fermionic holon operator, and ``bꜛ``, ``bꜜ`` are bosonic spinon operators.
""" transform_slave_fermion
function transform_slave_fermion(O::AbstractTensorMap)
    (N = numin(O)) == numout(O) || throw(ArgumentError("not a valid operator"))
    aux_charge = slave_fermion_auxiliary_charge(sectortype(O))
    aux_space = spacetype(O)(aux_charge => 1)
    aux_operator = id(Int, aux_space^N)

    return fuse_local_operators(O, aux_operator)
end
function transform_slave_fermion(V::ElementarySpace)
    charge = slave_fermion_auxiliary_charge(sectortype(V))
    V_aux = spacetype(V)(charge => 1)
    return fuse(V, V_aux)
end

end
