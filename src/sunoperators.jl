module SUNOperators

using TensorKit

export sun_space, basis_transform
export exchange, swap, biquadratic

const _ext_error = ErrorException(
    "SUNOperators requires SUNRepresentations.jl to be loaded. " *
        "Add `using SUNRepresentations` before using this module."
)

"""
    sun_space([symmetry::Type]; irrep)

Return the local Hilbert space for a single SU(N) site in the given irrep.

`irrep` is forwarded directly to the `SUNIrrep` constructor, which accepts weight tuples
(the number of entries fixes `N`), Dynkin-label tuples, or an integer dimension. For
example, `irrep=(1, 0)` selects the SU(2) fundamental (spin-½) and `irrep=(1, 0, 0)` the
SU(3) fundamental.

| Symmetry    | Space |
|-------------|-------|
| `SUNIrrep`  | single-sector `GradedSpace` (requires SUNRepresentations.jl) |
| `Trivial`   | `ComplexSpace(dim(irrep))` (requires SUNRepresentations.jl) |
"""
sun_space(args...; kwargs...) = throw(_ext_error)

"""
    basis_transform(symmetry::Type{<:Sector}; irrep)

Return the basis transformation that maps the dense basis of the `Trivial` operators onto
the basis of `sun_space(symmetry; irrep)`, as a `TensorMap` over `ComplexSpace`s.

SUNRepresentations.jl's generator matrices and TensorKit's SU(N) fusion tensors both use the
Gelfand–Tsetlin basis, so this transformation is the identity. It is exposed for use with
[`symmetrize`](@ref TensorKitTensors.symmetrize) and for testing.

Requires SUNRepresentations.jl to be loaded.
"""
basis_transform(args...; kwargs...) = throw(_ext_error)

"""
    exchange([eltype], [symmetry]; irrep)
    exchange([eltype], [symmetry]; irreps=(irrep1, irrep2))

The SU(N) two-site exchange (spin-exchange) operator

```math
P_{\\text{ex}} = \\sum_a T^a \\otimes T^a ,
```

the polarized quadratic Casimir built from the SU(N) generators ``T^a``.
It is the `Trivial` (dense) operator that becomes block diagonal under `SUNIrrep`, with eigenvalue
``\\tfrac{1}{2}[C₂(R) - C₂(λ₁) - C₂(λ₂)]`` on each coupled sector `R`.

`irrep` (or `irreps`) is forwarded to the `SUNIrrep` constructor.
Use `irreps` to couple two different SU(N) irreps.

Default element type: `ComplexF64`. Default symmetry: `Trivial`.

Supported symmetries: `Trivial`, `SUNIrrep` (requires SUNRepresentations.jl).
"""
exchange(args...; kwargs...) = throw(_ext_error)

"""
    swap([eltype], [symmetry]; irrep)

The SU(N) swap operator, i.e. the permutation ``P\\,|i⟩|j⟩ = |j⟩|i⟩`` of two sites carrying
the same irrep. For the fundamental representation it satisfies

```math
\\text{swap} = 2 ⋅ \\text{exchange} + \\tfrac{1}{N}\\,\\mathbf{1} .
```

`irrep` is forwarded to the `SUNIrrep` constructor.

Default element type: `ComplexF64`. Default symmetry: `Trivial`.

Supported symmetries: `Trivial`, `SUNIrrep` (requires SUNRepresentations.jl).
"""
swap(args...; kwargs...) = throw(_ext_error)

"""
    biquadratic([eltype], [symmetry]; irrep)
    biquadratic([eltype], [symmetry]; irreps=(irrep1, irrep2))

The SU(N) two-site biquadratic exchange operator ``P_{\\text{ex}}^2``, i.e. the square of
[`exchange`](@ref). Together with `exchange` it makes up the bilinear-biquadratic family
``J_1\\,P_{\\text{ex}} + J_2\\,P_{\\text{ex}}^2``.

`irrep` (or `irreps`) is forwarded to the `SUNIrrep` constructor.

Default element type: `ComplexF64`. Default symmetry: `Trivial`.

Supported symmetries: `Trivial`, `SUNIrrep` (requires SUNRepresentations.jl).
"""
biquadratic(args...; kwargs...) = throw(_ext_error)

end
