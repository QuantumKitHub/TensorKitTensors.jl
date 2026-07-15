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

    # only fall back to an identity seed for the empty-product case: seeding a non-empty
    # `reduce` forces an extra `⊗`, which widens the scalar type of the basis-transform
    # product (e.g. `RationalRoot{Int}` → `Float64`) and demotes the operator's precision
    Ucod = isempty(Us) ? id(Bool, one(ComplexSpace)) : reduce(⊗, Us)
    Udom = isempty(Uds) ? id(Bool, one(ComplexSpace)) : reduce(⊗, Uds)
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
function fuse_local_operators(O₁::AbstractTensorMap{<:Any, S₁, N₁, N₂}, O₂::AbstractTensorMap{<:Any, S₂, N₃, N₄}) where {S₁, S₂, N₁, N₂, N₃, N₄}
    S₁ == S₂ || throw(ArgumentError("operators have incompatible space types"))
    (N₁ == N₂ == N₃ == N₄) || throw(ArgumentError("operators have incompatible number of indices"))
    fuser = mapreduce(⊗, ntuple(identity, Val(N₁))) do i
        Vᵢ = space(O₁, i)
        Wᵢ = space(O₂, i)
        VWᵢ = fuse(Vᵢ, Wᵢ)
        return isomorphism(VWᵢ ← Vᵢ ⊗ Wᵢ)
    end
    O₁₂ = permute(
        O₁ ⊗ O₂, (
            ntuple(i -> iseven(i) ? N₁ + (i ÷ 2) : (i + 1) ÷ 2, Val(2N₁)),
            ntuple(i -> iseven(i) ? 3N₁ + (i ÷ 2) : 2N₁ + (i + 1) ÷ 2, Val(2N₁)),
        )
    )
    return fuser * O₁₂ * fuser'
end

# type annotation of a positional argument, from either `name::Type` or `::Type`
function _operator_type_arg(a)
    Meta.isexpr(a, :(::)) ||
        throw(Meta.ParseError("`@operator`: positional arguments must be annotated as `name::Type` or `::Type`"))
    return a.args[end]
end

# Build the block of forwarding methods (and optional alias) from the parsed reference
# definition. Returned unescaped; the macro escapes it.
function _operator_defs(alias, def, source)
    Meta.isexpr(def, :function) ||
        throw(Meta.ParseError("`@operator` expects `[alias] function op(...) ... end`"))
    sig = def.args[1]
    Meta.isexpr(sig, :call) ||
        throw(Meta.ParseError("`@operator`: the wrapped definition must be a function call signature"))
    fname = sig.args[1]
    (fname isa Symbol) ||
        throw(ArgumentError("`@operator`: the operator name must be a symbol"))

    posargs = filter(a -> !Meta.isexpr(a, :parameters), sig.args[2:end])
    length(posargs) >= 1 ||
        throw(ArgumentError("`@operator` expects at least an element type argument"))

    eltconstraint = _operator_type_arg(posargs[1])
    N = length(posargs) - 1  # number of symmetry arguments; may be zero

    loc = LineNumberNode(source.line, source.file)
    # every generated method takes and forwards `kwargs...` verbatim; only the wrapped
    # reference method validates keyword names, defaults, and types. signatures and calls
    # share the same `f(args...; kwargs...)` shape, so a single builder covers both.
    kwargs = gensym(:kwargs)
    call_expr(f, cargs...) = Expr(:call, f, Expr(:parameters, Expr(:..., kwargs)), cargs...)
    method_def(msig, body) = Expr(:function, msig, Expr(:block, loc, body))

    # `op(; kwargs...) = op(ComplexF64, Trivial, …; kwargs...)` — always emitted; fills the
    # element type default (and any symmetry defaults). This is the doc'd method.
    method_none = method_def(
        call_expr(fname),
        call_expr(fname, :ComplexF64, ntuple(Returns(:Trivial), N)...)
    )

    # With no symmetry arguments the wrapped reference method is itself the terminal, so
    # `method_none` (plus the reference's own `op(elt; …)`) is all that is needed. With one
    # or more symmetries, symmetries are all-or-nothing and we additionally emit the
    # symmetry-first, elt-only, and symmetric-terminal methods.
    extra_methods = if N == 0
        ()
    else
        # `op(s₁::Type{<:Sector}, …, sₙ::Type{<:Sector}; kwargs...) =
        #      op(ComplexF64, s₁, …, sₙ; kwargs...)`
        snames = [gensym(:S) for _ in 1:N]
        symsig = call_expr(
            fname, (Expr(:(::), snames[i], :(Type{<:Sector})) for i in 1:N)...
        )
        symcall = call_expr(fname, :ComplexF64, snames...)
        method_allsyms = method_def(symsig, symcall)

        # `op(elt::<eltconstraint>; kwargs...) = op(elt, Trivial, …; kwargs...)`
        eltname = gensym(:elt)
        eltsig = call_expr(fname, Expr(:(::), eltname, eltconstraint))
        eltcall = call_expr(fname, eltname, ntuple(Returns(:Trivial), N)...)
        method_elt = method_def(eltsig, eltcall)

        # symmetric terminal:
        # `op(elt::<c>, s₁::Type{<:Sector}, …; kwargs...) =
        #      _symmetrize_operator(op(elt, Trivial, …; kwargs...), s₁, …; kwargs...)`
        tname = gensym(:elt)
        tsyms = [gensym(:S) for _ in 1:N]
        tsig = call_expr(
            fname, Expr(:(::), tname, eltconstraint),
            (Expr(:(::), tsyms[i], :(Type{<:Sector})) for i in 1:N)...
        )
        refcall = call_expr(fname, tname, ntuple(Returns(:Trivial), N)...)
        hookcall = call_expr(:_symmetrize_operator, refcall, tsyms...)
        method_terminal = method_def(tsig, hookcall)

        (method_allsyms, method_elt, method_terminal)
    end

    aliasdef = alias === nothing ? nothing : Expr(:const, Expr(:(=), alias, fname))

    return quote
        $def
        Core.@__doc__ $method_none
        $(extra_methods...)
        $loc
        $aliasdef
    end
end

"""
    @operator [alias] function op(elt::Type{<:Number}[, ::Type{Trivial}, …]; kwargs...)
        ...
    end

Generate the public default-argument interface for an operator from its non-symmetric
reference implementation. The wrapped method is the single source of truth for the operator's
element type constraint, number of symmetry slots, keyword names, keyword defaults, and
keyword type annotations.

The wrapped signature must have an element type as its first positional argument, optionally
followed by one or more symmetry arguments (conventionally annotated `::Type{Trivial}`).
`@operator` emits the boilerplate so that all of the following resolve:

- `op()`
- `op(eltype)`
- `op(symmetry₁, …, symmetryₙ)` (all symmetries)
- `op(eltype, symmetry₁, …, symmetryₙ)`

The element type defaults to `ComplexF64`, and omitted symmetries default to `Trivial`.
Symmetries are all-or-nothing: either all of them are given, or none are. This avoids silently
defaulting the unspecified symmetries of a multi-symmetry operator.

If the reference method takes no symmetry arguments, it is itself the terminal: only the
`op()` element-type default is generated (the symmetry-first, elt-only, and symmetric-terminal
methods, including the `_symmetrize_operator` hook, are omitted).

Otherwise, the last generated method, the *symmetric terminal*, delegates to the module-local
hook
`_symmetrize_operator(op(elt, Trivial, …; kwargs...), symmetry₁, …; kwargs...)`, which each
operator module defines once (typically wrapping [`symmetrize`](@ref) with the module's
`basis_transform` and local space). Every generated method simply accepts `kwargs...` and
forwards them verbatim to both the reference call and the hook; the wrapped reference method
is the only place that validates keyword names, defaults, and types.

If an `alias` symbol is supplied, a `const alias = op` binding is emitted as well (e.g. the
Unicode name `Sᶻ` for `S_z`).

The first generated method is wrapped in `Core.@__doc__`, so a docstring written directly
above the `@operator` line is attached to the operator function:

```julia
\"\"\"
    S_z([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin-z operator.
\"\"\"
@operator Sᶻ function S_z(elt::Type{<:Number}, ::Type{Trivial}; spin = 1 // 2)
    ...
end
```
"""
macro operator(args...)
    (1 <= length(args) <= 2) ||
        throw(Meta.ParseError("`@operator` expects `[alias] function op(...) ... end`"))
    if length(args) == 2
        alias, def = args[1], args[2]
        (alias isa Symbol) ||
            throw(ArgumentError("`@operator`: the alias must be a symbol"))
    else
        alias, def = nothing, args[1]
    end
    return esc(_operator_defs(alias, def, __source__))
end
