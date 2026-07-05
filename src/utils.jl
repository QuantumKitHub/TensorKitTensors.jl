"""
    symmetrize(O::AbstractTensorMap, U::AbstractMatrix, V::ElementarySpace; tol=..., name="operator")
    symmetrize(O::AbstractTensorMap, Us::NTuple{N,AbstractMatrix}, V::ElementarySpace; tol=..., name="operator")

Construct the symmetric version of an ``N``-site operator `O` on the space `V^N ← V^N`,
given the basis transformation `U` that maps the basis of `O` onto the basis of `V`.

The operator `O` is first converted to a dense array, then rotated by applying `U` to every
site (or `Us[i]` to site `i`), and finally projected onto the symmetric tensor structure of
`V^N ← V^N` using the `TensorMap` constructor. If the rotated operator is not symmetric,
i.e. if it has nonzero entries (larger than `tol`) that are incompatible with the symmetry
structure of `V`, an `ArgumentError` is thrown mentioning `name`.

The default `tol` is `sqrt(eps)` of the scalar type of the rotated operator, floored at
`sqrt(eps(Float64))`. The floor reflects that the fusion-tensor data used by the projection
is computed at `Float64` precision for non-abelian sectors: abelian symmetries (`Z2Irrep`,
`U1Irrep`, `FermionParity`, and their products) preserve the full precision of the input,
while for non-abelian sectors (e.g. `SU2Irrep`) the result of wider scalar types such as
`BigFloat` is only accurate up to `Float64` precision.

Each basis transformation `Us[i]` should be a unitary matrix whose *rows* are indexed by the
basis of `V`, in the order defined by TensorKit (grouped per sector, in the order of
`sectors(V)`), and whose *columns* are indexed by the basis of `space(O, i)` in that same
convention. In other words, the dense representation of the symmetrized operator is
``(U_1 ⊗ ⋯ ⊗ U_N) O (U_1 ⊗ ⋯ ⊗ U_N)^†``.

The basis transformations of the operator modules in this package are documented and exposed
through their respective `basis_transform` functions.

# Examples

Symmetrizing the transverse-field term of the Ising model with respect to its ``ℤ₂``
spin-flip symmetry, using the Hadamard transformation to map the ``S^z`` basis onto the
``S^x`` basis:

```jldoctest
julia> using TensorKit, TensorKitTensors, TensorKitTensors.SpinOperators;

julia> X = S_x(); # single-site trivial operator

julia> U = basis_transform(Z2Irrep); # Hadamard matrix

julia> X_z2 = symmetrize(X, U, spin_space(Z2Irrep); name = "S_x");

julia> block(X_z2, Z2Irrep(0)) ≈ fill(1 / 2, 1, 1) && block(X_z2, Z2Irrep(1)) ≈ fill(-1 / 2, 1, 1)
true
```
"""
function symmetrize(
        O::AbstractTensorMap, Us::Tuple{Vararg{AbstractMatrix, N}}, V::ElementarySpace;
        tol = nothing, name = "operator"
    ) where {N}
    numout(O) == numin(O) == N ||
        throw(ArgumentError("number of basis transformations does not match the number of sites"))

    A = convert(Array, O)
    D = prod(ntuple(i -> size(A, i), N))
    U = reduce(kron, reverse(Us)) # column-major: first site corresponds to fastest index
    B = U * reshape(A, D, D) * U'
    # the default tolerance is floored at the Float64 resolution, since the fusion-tensor
    # data used by the projection is itself computed at Float64 precision for non-abelian
    # sectors
    tol′ = something(tol, sqrt(max(eps(real(float(eltype(B)))), eps(Float64))))
    P = ProductSpace(ntuple(Returns(V), Val(N))...)
    return try
        TensorMap(B, P ← P; tol = tol′)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("`$name` is not symmetric under `$(sectortype(V))` symmetry"))
    end
end
function symmetrize(O::AbstractTensorMap, U::AbstractMatrix, V::ElementarySpace; kwargs...)
    return symmetrize(O, ntuple(Returns(U), numout(O)), V; kwargs...)
end

"""
    _restrict_scalartype(T::Type{<:Number}, t::AbstractTensorMap; name="operator")

Return a copy of `t` with scalar type `T`, or `t` itself if it already has scalar type `T`.
Throws an `ArgumentError` mentioning `name` if `T <: Real` while `t` has entries with a
nonzero imaginary part.
"""
function _restrict_scalartype(::Type{T}, t::AbstractTensorMap; name = "operator") where {T <: Number}
    scalartype(t) === T && return t
    if T <: Real && !(scalartype(t) <: Real)
        ε = sqrt(eps(real(float(scalartype(t)))))
        for (_, b) in blocks(t)
            all(x -> abs(imag(x)) <= ε * max(one(ε), abs(x)), b) ||
                throw(ArgumentError("`$name` requires a complex scalar type, got `$T`"))
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
