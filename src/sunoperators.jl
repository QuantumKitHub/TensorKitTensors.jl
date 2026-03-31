module SUNOperators

using TensorKit

export sun_space
export exchange, swap, twosite_casimir

const _ext_error = ErrorException(
    "SUNOperators requires SUNRepresentations.jl to be loaded. " *
        "Add `using SUNRepresentations` before using this module."
)

"""
    sun_space([symmetry::Type]; irrep)

Return the local Hilbert space for a single SU(N) site in the given irrep.

`irrep` is forwarded directly to the `SUNIrrep` constructor, which accepts
Dynkin-label tuples, weight tuples, or an integer dimension; N is inferred
automatically. For example, `irrep=(1,)` selects the SU(2) fundamental (spin-½).

Supported symmetries:

| Symmetry    | Space |
|-------------|-------|
| `SUNIrrep`  | single-sector `GradedSpace` (requires SUNRepresentations.jl) |
| `Trivial`   | `ComplexSpace(dim(irrep))` (requires SUNRepresentations.jl) |
"""
sun_space(args...; kwargs...) = throw(_ext_error)

"""
    exchange([eltype], [symmetry]; irrep)
    exchange([eltype], [symmetry]; irreps=(irrep1, irrep2))

The SU(N) exchange operator

```math
P_{\\text{ex}} = \\tfrac{1}{2}\\bigl[C_2(R_{12}) - C_2(R_1) - C_2(R_2)\\bigr]
```

diagonal in the fusion basis with eigenvalue
``\\tfrac{1}{2}[C_2(R) - C_2(\\lambda_1) - C_2(\\lambda_2)]``
for each coupled sector `R`.

`irrep` (or `irreps`) is forwarded to the `SUNIrrep` constructor and accepts
Dynkin-label tuples, weight tuples, or an integer dimension. Use `irreps` to
couple two different SU(N) irreps.

Default element type: `Float64`. Default symmetry: `SUNIrrep`.

Supported symmetries: `SUNIrrep`, `Trivial`.
"""
exchange(args...; kwargs...) = throw(_ext_error)

"""
    swap([eltype], [symmetry]; irrep)

The SU(N) swap (permutation) operator satisfying

```math
\\text{swap} = 2 \\cdot \\text{exchange} + \\tfrac{1}{N} \\mathbf{1}
```

`irrep` is forwarded to the `SUNIrrep` constructor. Supported symmetries: `SUNIrrep`, `Trivial`.
"""
swap(args...; kwargs...) = throw(_ext_error)

"""
    twosite_casimir([eltype], [symmetry]; k=2, irrep)
    twosite_casimir([eltype], [symmetry]; k=2, irreps=(irrep1, irrep2))

Two-site operator diagonal in the fusion basis with eigenvalue ``C_k(R)``
for each coupled sector `R`, where ``C_k`` is the `k`-th Casimir of SU(N).

`irrep` (or `irreps`) is forwarded to the `SUNIrrep` constructor. Default `k=2`.

Supported symmetries: `SUNIrrep`, `Trivial`.
"""
twosite_casimir(args...; kwargs...) = throw(_ext_error)

end
