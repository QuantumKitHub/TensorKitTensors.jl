module TJOperators

using LinearAlgebra: diagind
using TensorKit
import ..HubbardOperators
import ..TensorKitTensors: symmetrize, desymmetrize, fuse_charge, @operator

export tj_space, basis_transform, symmetrize_operator, tj_projector
export transform_slave_fermion
# the operator names and their aliases are exported by the generation loop at the bottom of
# this file, from the `_OPERATORS` registry

const _docs_basis_table = """
```
| label | tJ basis | slave-fermion |
| ----- | -------- | ------------- |
|  |0⟩  |   |∅⟩    |      h⁺|∅⟩    |
|  |↑⟩  |  u⁺|∅⟩   |     bꜛ⁺|∅⟩    |
|  |↓⟩  |  d⁺|∅⟩   |     bꜜ⁺|∅⟩    |
```
"""

# Spaces
# ------
@doc """
    tj_space([particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)

Return the local hilbert space for a t-J-type model with the given particle and spin symmetries.
The basis consists of the following states:

$_docs_basis_table

- `|∅⟩` is the vacuum state;
- `u` and `d` denote fermionic spin-up and spin-down operators;
- in the slave-fermion representation, ``h`` is the fermionic holon operator, and `bꜛ`, `bꜜ` are bosonic spinon operators.

The possible symmetries are:
- Particle number : `Trivial`, `U1Irrep`
- Spin            : `Trivial`, `U1Irrep`, `SU2Irrep`.

Use `slave_fermion = true` to switch to the slave-fermion basis, which flips the fermion
parity of every state; see [`transform_slave_fermion`](@ref).
""" tj_space
function tj_space(
        particle_symmetry::Type{<:Sector} = Trivial,
        spin_symmetry::Type{<:Sector} = Trivial;
        slave_fermion::Bool = false
    )
    V = _tj_space(particle_symmetry, spin_symmetry)
    return slave_fermion ? transform_slave_fermion(V) : V
end

# the t-J space in the plain basis; dispatched on the symmetries, with the `slave_fermion`
# keyword handled once by `tj_space` above
_tj_space(::Type{Trivial}, ::Type{Trivial}) = Vect[FermionParity](0 => 1, 1 => 2)
function _tj_space(::Type{Trivial}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
end
function _tj_space(::Type{Trivial}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ SU2Irrep]((0, 0) => 1, (1, 1 // 2) => 1)
end
function _tj_space(::Type{U1Irrep}, ::Type{Trivial})
    return Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 2)
end
function _tj_space(::Type{U1Irrep}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep](
        (0, 0, 0) => 1, (1, 1, 1 // 2) => 1, (1, 1, -1 // 2) => 1
    )
end
function _tj_space(::Type{U1Irrep}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1)
end
# the η-pairing doublet of the Hubbard model is (|↑↓⟩, |0⟩), which has no t-J counterpart
function _tj_space(::Type{SU2Irrep}, spin_symmetry::Type{<:Sector})
    throw(ArgumentError("t-J model does not have ``SU(2)`` particle symmetry."))
end
function _tj_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    throw(ArgumentError("invalid symmetry `($particle_symmetry, $spin_symmetry)`"))
end

"""
    basis_transform(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the unitary basis transformation that maps the basis of
`tj_space(Trivial, Trivial; slave_fermion)` onto the basis of
`tj_space(particle_symmetry, spin_symmetry; slave_fermion)`, as a `TensorMap` between the
desymmetrized versions of these spaces (see
[`desymmetrize`](@ref TensorKitTensors.desymmetrize)), as required by
[`symmetrize`](@ref TensorKitTensors.symmetrize).

For all symmetry combinations the transformation is a permutation, determined by the sector
order of the target space, where the states are identified as follows:

- For `U1Irrep` particle symmetry, the number of electrons is used as charge, distinguishing
  ``|0⟩`` (charge 0) from ``|↑⟩`` and ``|↓⟩`` (charge 1).
- For `U1Irrep` spin symmetry, the ``S^z`` eigenvalue ``±1/2`` is used as charge,
  distinguishing ``|↑⟩`` from ``|↓⟩``.
- For `SU2Irrep` spin symmetry, ``(|↑⟩, |↓⟩)`` forms the spin doublet (descending ``m``).

Both bases order the states by fermion parity, so the reference basis differs between them:

```
| basis          | reference order |
| -------------- | --------------- |
| t-J            | |0⟩, |↑⟩, |↓⟩   |
| slave-fermion  | |↑⟩, |↓⟩, |0⟩   |
```

The transformations have exact integer entries and are therefore returned with integer
scalar type, such that they promote to any scalar type without loss of precision.
"""
function basis_transform(
        particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
        slave_fermion::Bool = false
    )
    reference = _dense_state_order(Trivial, Trivial; slave_fermion)
    U = zeros(Int, 3, 3)
    for (row, state) in enumerate(_dense_state_order(particle_symmetry, spin_symmetry; slave_fermion))
        U[row, findfirst(==(state), reference)] = 1
    end
    V = tj_space(particle_symmetry, spin_symmetry; slave_fermion)
    return TensorMap(U, desymmetrize(V) ← desymmetrize(tj_space(Trivial, Trivial; slave_fermion)))
end

# the states (|0⟩, |↑⟩, |↓⟩) = (1, 2, 3) in the dense order of
# `tj_space(particle_symmetry, spin_symmetry; slave_fermion)`
function _dense_state_order(
        particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
        slave_fermion::Bool = false
    )
    V = tj_space(particle_symmetry, spin_symmetry; slave_fermion)
    states = Int[]
    for c in sectors(V)
        # a one-dimensional auxiliary space maps `c → c ⊠ aux` while preserving the
        # multiplicity order, so fusing in the auxiliary charge again (it is self-inverse)
        # recovers the plain-basis sector that identifies the states
        c′ = slave_fermion ? only(c ⊗ slave_fermion_auxiliary_charge(typeof(c))) : c
        append!(states, _state_indices(c′, particle_symmetry, spin_symmetry))
    end
    return states
end

# t-J basis indices (|0⟩, |↑⟩, |↓⟩) = (1, 2, 3) contained in sector `c`, in dense row order
function _state_indices(c, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    parity = c isa FermionParity ? c : c[1]
    return if !parity.isodd # parity even: |0⟩
        (1,)
    elseif spin_symmetry === Trivial || spin_symmetry === SU2Irrep
        (2, 3)
    else # U1Irrep: Sᶻ distinguishes the states (spin is the last sector factor)
        spin_sector = particle_symmetry === Trivial ? c[2] : c[3]
        spin_sector.charge > 0 ? (2,) : (3,)
    end
end

"""
    symmetrize_operator(O::AbstractTensorMap, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false, tol=nothing)

Symmetrize a t-J operator defined on `tj_space(Trivial, Trivial; slave_fermion)` through the basis transformation for the requested particle and spin symmetries.
The input must already use the normal or slave-fermion representation selected by `slave_fermion`.
"""
function symmetrize_operator(
        O::AbstractTensorMap, particle_symmetry::Type{<:Sector},
        spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false, tol = nothing
    )
    return symmetrize(
        O, basis_transform(particle_symmetry, spin_symmetry; slave_fermion),
        tj_space(particle_symmetry, spin_symmetry; slave_fermion); tol
    )
end

# Relation to the Hubbard model
# -----------------------------
"""
    tj_projector(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Projection operator from Hubbard space to t-J space, which removes the doubly occupied state
``|↑↓⟩``. The operators of this module are *defined* as the projections of their
`HubbardOperators` counterparts of the same name, i.e. they satisfy

```julia
proj = reduce(⊗, ntuple(Returns(tj_projector(P, S)), N))
TJOperators.op(elt, P, S) ≈ proj * HubbardOperators.op(elt, P, S) * proj'
```

for an `N`-site operator `op`. The double-occupancy operators of the Hubbard model
(`ud_num` and `half_ud_num`) have no t-J counterpart, as they project to zero.

The projector is defined in the plain t-J basis only, since the slave-fermion basis has no
Hubbard counterpart. It is an isometry from the four-dimensional Hubbard space onto the
three-dimensional t-J space, so `proj * proj' == id(tj_space(P, S))` while `proj' * proj` is the
projector within the Hubbard space.

The scalartype is `Int` to avoid floating point errors.

Supported symmetries: particle `Trivial`, `U1Irrep`; spin `Trivial`, `U1Irrep`, `SU2Irrep`.
"""
function tj_projector(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    Vhub = HubbardOperators.hubbard_space(particle_symmetry, spin_symmetry)
    VtJ = tj_space(particle_symmetry, spin_symmetry)
    proj = zeros(Int, Vhub → VtJ)

    # the retained states are the leading rows of every shared sector: for `Trivial` particle
    # symmetry `|↑↓⟩` is the *second* even state of the Hubbard space, and for `U1Irrep` it
    # lives in a sector that has no t-J counterpart at all, so no block of `proj` contains it
    for (f1, f2) in fusiontrees(proj)
        proj[f1, f2][diagind(proj[f1, f2])] .= 1
    end
    return proj
end

"Project a Hubbard reference operator onto the normal t-J space."
function _project_hubbard(O::AbstractTensorMap)
    # The projector is parity-even, so it distributes over `⊗`
    n = numout(O)
    Pⁿ = reduce(⊗, ntuple(Returns(tj_projector(Trivial, Trivial)), Val(n)))
    # integer entries in `Pⁿ` preserve the scalar type of `O`
    return Pⁿ * O * Pⁿ'
end

# Slave-fermion basis
# -------------------
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

where ``h`` is the fermionic holon operator, and `bꜛ`, `bꜜ` are bosonic spinon operators.

Fusing in the auxiliary fermionic charge flips the parity of every state, which changes the
statistics of the operator: braiding the auxiliary legs of an ``N``-site operator through the
physical ones generates a staggered sign ``(-1)^{(k-1)p_k}`` on site ``k``, with ``p_k`` the
parity of the state. Consequently this transformation has to be applied to a complete
operator, and does not commute with taking tensor products of single-site operators.
""" transform_slave_fermion
function transform_slave_fermion(O::AbstractTensorMap)
    (N = numin(O)) == numout(O) || throw(ArgumentError("not a valid operator"))
    aux_charge = slave_fermion_auxiliary_charge(sectortype(O))
    return fuse_charge(O, aux_charge)
end
function transform_slave_fermion(V::ElementarySpace)
    charge = slave_fermion_auxiliary_charge(sectortype(V))
    V_aux = spacetype(V)(charge => 1)
    return fuse(V, V_aux)
end

# Operators
# ---------

# Keeps the prose of a Hubbard docstring, regenerate the signature block with `slave_fermion`
# drop admonitions (by convention Hubbard-only)
function _operator_docstring(name::Symbol, alias::Symbol, hubbard_doc)
    _OPERATOR_ARGS = "([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}]; slave_fermion::Bool = false)"
    return string(
        "    ", name, _OPERATOR_ARGS, "\n",
        "    ", alias, _OPERATOR_ARGS, "\n\n",
        _project_symmetries(_inherit_description(_docstring_text(hubbard_doc))), "\n\n",
        "This operator is the projection of [`HubbardOperators.", name,
        "`](@ref HubbardOperators.", name, ") onto the t-J space, see [`tj_projector`](@ref). ",
        "Use `slave_fermion = true` to obtain it in the slave-fermion basis, see ",
        "[`transform_slave_fermion`](@ref).\n",
    )
end

# `@doc` hands back the raw `DocStr` on recent Julia versions, and a rendered `Markdown.MD`
# object on older ones; stringifying the latter reflows the text and writes inline math as
# `$x$` instead of ``x``, which parses the same way.
_docstring_text(doc::Base.Docs.DocStr) = join(doc.text)
_docstring_text(doc) = string(doc)

_is_toplevel(line) = !isempty(strip(line)) && !startswith(line, ' ')
function _inherit_description(docstring::AbstractString)
    lines = split(docstring, '\n')
    i = if startswith(first(lines), "```")
        closing = findnext(startswith("```"), lines, firstindex(lines) + 1)
        something(closing, lastindex(lines)) + 1
    else
        something(findfirst(_is_toplevel, lines), lastindex(lines) + 1)
    end
    description = String[]
    while i <= lastindex(lines)
        if startswith(lines[i], "!!! ")
            i += 1
            while i <= lastindex(lines) && !_is_toplevel(lines[i])
                i += 1
            end
        else
            push!(description, lines[i])
            i += 1
        end
    end
    return strip(join(description, '\n'))
end

# The t-J space has no ``SU(2)`` particle symmetry -- the η-pairing doublet of the Hubbard model
# is (|↑↓⟩, |0⟩), whose doubly occupied state is projected out -- so an inherited symmetry
# listing has to drop it from the particle symmetries. That is the only difference: for every
# operator of the registry the spin symmetries carry over verbatim, and so do the remaining
# particle ones. The rendered docstring of Julia 1.10 may reflow the listing onto several lines,
# hence matching across whitespace; the particle symmetries are the ones terminated by `;`.
function _project_symmetries(description::AbstractString)
    return replace(
        description, r"(symmetries:.*?),\s*`SU2Irrep`(\s*);"s => s"\1\2;"
    )
end

for (name, alias) in [
        # single-site operators
        (:u_num, :nꜛ), (:d_num, :nꜜ), (:e_num, :n), (:h_num, :nʰ),
        (:S_plus, :S⁺), (:S_min, :S⁻),
        (:S_x, :Sˣ), (:S_y, :Sʸ), (:S_z, :Sᶻ),
        # two-site operators
        (:u_plus_u_min, :u⁺u⁻), (:d_plus_d_min, :d⁺d⁻),
        (:u_min_u_plus, :u⁻u⁺), (:d_min_d_plus, :d⁻d⁺),
        (:e_plus_e_min, :e⁺e⁻), (:e_min_e_plus, :e⁻e⁺), (:e_hopping, :e_hop),
        (:u_min_d_min, :u⁻d⁻), (:u_plus_d_plus, :u⁺d⁺),
        (:d_min_u_min, :d⁻u⁻), (:d_plus_u_plus, :d⁺u⁺),
        (:u_min_u_min, :u⁻u⁻), (:u_plus_u_plus, :u⁺u⁺),
        (:d_min_d_min, :d⁻d⁻), (:d_plus_d_plus, :d⁺d⁺),
        (:singlet_plus, :singlet⁺), (:singlet_min, :singlet⁻),
        (:singlet_plus_singlet_min_3site, :Δ⁺ij_Δjk), (:singlet_plus_singlet_min_4site, :Δ⁺ij_Δkl),
        (:S_plus_S_min, :S⁺S⁻), (:S_min_S_plus, :S⁻S⁺), (:S_exchange, :SS),
    ]
    hubbard_doc = @eval @doc(HubbardOperators.$name)
    @eval export $name, $alias
    @eval @doc $(_operator_docstring(name, alias, hubbard_doc)) @operator $alias function $name(
            elt::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
            slave_fermion::Bool = false
        )
        O = _project_hubbard(HubbardOperators.$name(elt, Trivial, Trivial))
        return slave_fermion ? transform_slave_fermion(O) : O
    end
end

end
