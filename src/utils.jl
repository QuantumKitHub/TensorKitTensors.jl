"""
    desymmetrize(V::VectorSpace)
    desymmetrize(t::AbstractTensorMap)

Map a symmetric (graded) vector space or tensor onto its non-symmetric counterpart over `ComplexSpace`.

For an elementary space, this is `ComplexSpace(dim(V))` with the duality of `V` preserved; product and hom spaces are mapped elementwise.
For a tensor, the result is the `TensorMap` over the desymmetrized spaces holding the dense representation of `t`, in which the basis vectors of each space are grouped per sector, in the order of `sectors(V)`.
This is the inverse operation of [`symmetrize`](@ref) (with trivial basis transformations).

!!! note
    For tensors with fermionic gradings, the dense representation discards the fermionic statistics:
    the resulting purely bosonic tensor incorporates the sign conventions of TensorKit's fusion-tensor data,
    and is consistent with what [`symmetrize`](@ref) expects, but permuting its indices is no longer equivalent
    to permuting the indices of the original tensor.
"""
desymmetrize(V::ComplexSpace) = V
desymmetrize(V::ElementarySpace) = isdual(V) ? ComplexSpace(dim(V))' : ComplexSpace(dim(V))
desymmetrize(P::ProductSpace) = ProductSpace{ComplexSpace}(map(desymmetrize, P))
desymmetrize(W::HomSpace) = desymmetrize(codomain(W)) ← desymmetrize(domain(W))
function desymmetrize(t::AbstractTensorMap)
    spacetype(t) === ComplexSpace && return t
    # `fusiontensor` warns that dense arrays of fermionic tensors do not preserve the
    # categorical properties, but discarding the fermionic statistics is exactly the
    # documented behavior here, so logging is disabled for the conversion
    A = with_logger(() -> convert(Array, t), NullLogger())
    return TensorMap(A, desymmetrize(space(t)))
end

"""
    symmetrize(O::AbstractTensorMap, U::AbstractTensorMap, V::ElementarySpace; tol=...)
    symmetrize(O::AbstractTensorMap, Us::NTuple{N, AbstractTensorMap}, V::ElementarySpace; tol=...)
    symmetrize(O::AbstractTensorMap, (Us, Uds)::Tuple{NTuple{M, AbstractTensorMap}, NTuple{N, AbstractTensorMap}}, V::HomSpace; tol=...)

Construct the symmetric version of an ``M ← N`` operator `O` on the space `V`, given the
basis transformations that map the basis of `O` onto the basis of `V`.

The operator `O` is first brought to its dense form (see [`desymmetrize`](@ref)), then
rotated by applying the basis transformations to each of its legs, and finally projected onto
the symmetric tensor structure of `V`. If the rotated operator is not symmetric, i.e. if it
has nonzero entries (larger than `tol`) that are incompatible with the symmetry structure of
`V`, an `ArgumentError` is thrown.

The most general form takes `V::HomSpace` (`V_cod ← V_dom`) and a pair of tuples of basis
transformations `(Us, Uds)` for each of the spaces of `V`. In other words, the dense
representation of the symmetrized operator is ``(U_1 ⊗ ⋯ ⊗ U_M)\\, O\\, (Ũ_1 ⊗ ⋯ ⊗ Ũ_N)^†``.

For the common case of a square operator over a single space `V::ElementarySpace`, a single
transformation `U` (applied to every leg) or an `N`-tuple `Us` (one per site, applied to both
codomain and domain) suffices, and the target space `V^N ← V^N` is constructed automatically.

The default `tol` is `sqrt(eps)` of the scalar type of the rotated operator, floored at
`sqrt(eps)` of the `TensorKit.sectorscalartype` of the symmetry, i.e. the element type of
the fusion-tensor data used by the projection. Sectors with exact (integer) topological
data, such as `Z2Irrep`, `U1Irrep`, `FermionParity`, and their products, preserve the full
precision of the input, while for sectors with floating-point data (e.g. `SU2Irrep`, whose
Clebsch-Gordan coefficients are `Float64`) the result of wider scalar types such as
`BigFloat` is only accurate up to that precision.

The basis transformations of the operator modules in this package are documented and exposed
through their respective `basis_transform` functions.

# Examples

Symmetrizing the transverse-field term of the Ising model with respect to its ``ℤ₂``
spin-flip symmetry, using the Hadamard transformation to map the ``S^z`` basis onto the
``S^x`` basis:

```jldoctest; filter = r"([0-9]+[.][0-9]+?)(0{4,}[0-9]+)?" => s"\\1"
julia> using TensorKit, TensorKitTensors, TensorKitTensors.SpinOperators;

julia> X = S_x(); # single-site trivial operator

julia> U = basis_transform(Z2Irrep); # Hadamard transformation

julia> X_z2 = symmetrize(X, U, spin_space(Z2Irrep))
2←2 TensorMap{ComplexF64, Rep[ℤ₂], 1, 1, Vector{ComplexF64}}:
 codomain: ⊗(Rep[ℤ₂](0 => 1, 1 => 1))
 domain: ⊗(Rep[ℤ₂](0 => 1, 1 => 1))
 blocks:
 * Irrep[ℤ₂](0) => 1×1 reshape(view(::Vector{ComplexF64}, 1:1), 1, 1) with eltype ComplexF64:
 0.5 + 0.0im

 * Irrep[ℤ₂](1) => 1×1 reshape(view(::Vector{ComplexF64}, 2:2), 1, 1) with eltype ComplexF64:
 -0.5 + 0.0im
```
"""
function symmetrize(
        O::AbstractTensorMap,
        (Us, Uds)::Tuple{NTuple{M, AbstractTensorMap}, NTuple{N, AbstractTensorMap}},
        V::HomSpace;
        tol = nothing
    ) where {M, N}
    numout(O) == M && numin(O) == N ||
        throw(ArgumentError("number of basis transformations does not match the number of indices"))
    length(codomain(V)) == M && length(domain(V)) == N ||
        throw(ArgumentError("target space `V` does not match the number of indices"))

    Ucod = reduce(⊗, Us; init = id(one(ComplexSpace)))
    Udom = reduce(⊗, Uds; init = id(one(ComplexSpace)))
    B = Ucod * desymmetrize(O) * Udom'
    tol′ = something(tol, _default_tol(scalartype(B), sectortype(V)))
    return try
        # disable logging to suppress the fermionic `fusiontensor` warning, see `desymmetrize`
        with_logger(() -> TensorMap(convert(Array, B), V; tol = tol′), NullLogger())
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("operator is not symmetric under `$(sectortype(V))` symmetry"))
    end
end
function symmetrize(
        O::AbstractTensorMap, Us::NTuple{N, AbstractTensorMap}, V::ElementarySpace;
        kwargs...
    ) where {N}
    P = ProductSpace(ntuple(Returns(V), Val(N))...)
    return symmetrize(O, (Us, Us), P ← P; kwargs...)
end
symmetrize(O::AbstractTensorMap, U::AbstractTensorMap, V::ElementarySpace; kwargs...) =
    symmetrize(O, ntuple(Returns(U), numout(O)), V; kwargs...)

# `sqrt(eps)` of the scalar type, floored at the resolution of the sector scalar type,
# i.e. the element type of the fusion-tensor data used by the projection
function _default_tol(::Type{T}, I::Type{<:Sector}) where {T <: Number}
    ε = eps(real(float(T)))
    Tsector = real(TensorKit.sectorscalartype(I))
    return sqrt(Tsector <: AbstractFloat ? max(ε, eps(Tsector)) : ε)
end

"""
    fuse_local_operators(O₁, O₂)

Given two ``n``-body operators, acting on ``ℋ₁ = V₁ ⊗ ⋯ ⊗ Vₙ`` and ``ℋ₂ = W₁ ⊗ ⋯ ⊗ Wₙ``,
return the operator acting on the fused local spaces, i.e. on ``ℋ = fuse(V₁ ⊗ W₁) ⊗ ⋯ ⊗ fuse(Vₙ ⊗ Wₙ)``.
"""
function fuse_local_operators(O₁::AbstractTensorMap, O₂::AbstractTensorMap)
    spacetype(O₁) == spacetype(O₂) ||
        throw(ArgumentError("operators have incompatible space types"))
    (N = numout(O₁)) == numin(O₁) == numout(O₂) == numin(O₂) ||
        throw(ArgumentError("operators have incompatible number of indices"))

    fuser = mapreduce(⊗, 1:N) do i
        Vᵢ = space(O₁, i)
        Wᵢ = space(O₂, i)
        VWᵢ = fuse(Vᵢ, Wᵢ)
        return isomorphism(VWᵢ ← Vᵢ ⊗ Wᵢ)
    end

    O₁₂ = permute(
        O₁ ⊗ O₂, (
            ntuple(i -> iseven(i) ? N + (i ÷ 2) : (i + 1) ÷ 2, 2N),
            ntuple(i -> iseven(i) ? 3N + (i ÷ 2) : 2N + (i + 1) ÷ 2, 2N),
        )
    )

    return fuser * O₁₂ * fuser'
end
