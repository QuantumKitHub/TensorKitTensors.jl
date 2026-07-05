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
desymmetrize(P::ProductSpace) = ProductSpace(map(desymmetrize, tuple(P...))...)
desymmetrize(W::HomSpace) = desymmetrize(codomain(W)) ← desymmetrize(domain(W))
function desymmetrize(t::AbstractTensorMap)
    spacetype(t) === ComplexSpace && return t
    return TensorMap(convert(Array, t), desymmetrize(space(t)))
end

"""
    symmetrize(O::AbstractTensorMap, U::AbstractTensorMap, V::ElementarySpace; tol=...)
    symmetrize(O::AbstractTensorMap, Us::NTuple{N,AbstractTensorMap}, V::ElementarySpace; tol=...)

Construct the symmetric version of an ``N``-site operator `O` on the space `V^N ← V^N`,
given the basis transformation `U` that maps the basis of `O` onto the basis of `V`.

The operator `O` is first brought to its dense form (see [`desymmetrize`](@ref)), then
rotated by applying `U` to every site (or `Us[i]` to site `i`), and finally projected onto
the symmetric tensor structure of `V^N ← V^N`. If the rotated operator is not symmetric,
i.e. if it has nonzero entries (larger than `tol`) that are incompatible with the symmetry
structure of `V`, an `ArgumentError` is thrown.

The default `tol` is `sqrt(eps)` of the scalar type of the rotated operator, floored at
`sqrt(eps)` of the `TensorKit.sectorscalartype` of the symmetry, i.e. the element type of
the fusion-tensor data used by the projection. Sectors with exact (integer) topological
data, such as `Z2Irrep`, `U1Irrep`, `FermionParity`, and their products, preserve the full
precision of the input, while for sectors with floating-point data (e.g. `SU2Irrep`, whose
Clebsch-Gordan coefficients are `Float64`) the result of wider scalar types such as
`BigFloat` is only accurate up to that precision.

Each basis transformation `Us[i]` should be a unitary `AbstractTensorMap` over `ComplexSpace`s,
with `desymmetrize(space(O, i))` as its domain and `desymmetrize(V)` as its codomain. In
other words, the dense representation of the symmetrized operator is
``(U_1 ⊗ ⋯ ⊗ U_N)\\, O\\, (U_1 ⊗ ⋯ ⊗ U_N)^†``.

The basis transformations of the operator modules in this package are documented and exposed
through their respective `basis_transform` functions.

# Examples

Symmetrizing the transverse-field term of the Ising model with respect to its ``ℤ₂``
spin-flip symmetry, using the Hadamard transformation to map the ``S^z`` basis onto the
``S^x`` basis:

```jldoctest
julia> using TensorKit, TensorKitTensors, TensorKitTensors.SpinOperators;

julia> X = S_x(); # single-site trivial operator

julia> U = basis_transform(Z2Irrep); # Hadamard transformation

julia> X_z2 = symmetrize(X, U, spin_space(Z2Irrep));

julia> block(X_z2, Z2Irrep(0)) ≈ fill(1 / 2, 1, 1) && block(X_z2, Z2Irrep(1)) ≈ fill(-1 / 2, 1, 1)
true
```
"""
function symmetrize(
        O::AbstractTensorMap, Us::Tuple{Vararg{AbstractTensorMap, N}}, V::ElementarySpace;
        tol = nothing
    ) where {N}
    numout(O) == numin(O) == N ||
        throw(ArgumentError("number of basis transformations does not match the number of sites"))

    U = reduce(⊗, Us)
    B = U * desymmetrize(O) * U'
    tol′ = something(tol, _default_tol(scalartype(B), sectortype(V)))
    P = ProductSpace(ntuple(Returns(V), Val(N))...)
    return try
        TensorMap(convert(Array, B), P ← P; tol = tol′)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("operator is not symmetric under `$(sectortype(V))` symmetry"))
    end
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
    _restrict_scalartype(T::Type{<:Number}, t::AbstractTensorMap)

Return a copy of `t` with scalar type `T`, or `t` itself if it already has scalar type `T`.
Throws an `ArgumentError` if `T <: Real` while `t` has entries with a nonzero imaginary
part.
"""
function _restrict_scalartype(::Type{T}, t::AbstractTensorMap) where {T <: Number}
    scalartype(t) === T && return t
    if T <: Real && !(scalartype(t) <: Real)
        ε = sqrt(eps(real(float(scalartype(t)))))
        for (_, b) in blocks(t)
            all(x -> abs(imag(x)) <= ε * max(one(ε), abs(x)), b) ||
                throw(ArgumentError("operator requires a complex scalar type, got `$T`"))
        end
    end
    tdst = similar(t, T)
    for (c, b) in blocks(t)
        block(tdst, c) .= T <: Real ? real.(b) : b
    end
    return tdst
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
