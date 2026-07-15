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

"""
    @operator [alias] function op(elt::Type{<:Number}, ::Type{Trivial}, …; kwargs...)
        ...
    end

Generate the public default-argument interface for an operator from its non-symmetric
reference implementation. The wrapped method is the single source of truth for the operator's
element type constraint, number of symmetry slots, keyword names, keyword defaults, and
keyword type annotations.

The wrapped signature must have an element type as its first positional argument, followed by
one or more concrete `::Type{Trivial}` symmetry arguments. `@operator` emits the boilerplate so
that all of the following resolve:

- `op()`
- `op(eltype)`
- `op(symmetry₁, …, symmetryₙ)` (all symmetries)
- `op(eltype, symmetry₁, …, symmetryₙ)`

The element type defaults to `ComplexF64`, and omitted symmetries default to `Trivial`.
Symmetries are all-or-nothing: either all of them are given, or none are. This avoids silently
defaulting the unspecified symmetries of a multi-symmetry operator.

The last generated method, the *symmetric terminal*, delegates to the module-local hook
`_symmetrize_operator(op(elt, Trivial, …; kwargs...), symmetry₁, …; kwargs...)`, which each
operator module defines once (typically wrapping [`symmetrize`](@ref) with the module's
`basis_transform` and local space). Keyword arguments are copied from the wrapped reference
method and forwarded by name to both the reference call and the hook. Vararg keywords
(`kwargs...`) are rejected because the public keyword interface is inferred from the concrete
method signature.

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
        error("`@operator` expects `[alias] function op(...) ... end`")
    if length(args) == 2
        alias, def = args[1], args[2]
        (alias isa Symbol) || error("`@operator`: the alias must be a symbol")
    else
        alias, def = nothing, args[1]
    end

    (def isa Expr && def.head === :function) ||
        error("`@operator` expects `[alias] function op(...) ... end`")
    sig = def.args[1]
    (sig isa Expr && sig.head === :call) ||
        error("`@operator`: the wrapped definition must be a function call signature")
    fname = sig.args[1]
    (fname isa Symbol) || error("`@operator`: the operator name must be a symbol")
    params = findfirst(a -> a isa Expr && a.head === :parameters, sig.args)
    kwtemplate = params === nothing ? Any[] : sig.args[params].args
    posargs = filter(a -> !(a isa Expr && a.head === :parameters), sig.args[2:end])
    length(posargs) >= 2 ||
        error("`@operator` expects an element type plus at least one symmetry argument")

    function parse_type_arg(a)
        (a isa Expr && a.head === :(::)) ||
            error("`@operator`: positional arguments must be annotated as `name::Type` or `::Type`")
        return a.args[end]
    end

    function is_trivial_type(t)
        return t isa Expr && t.head === :curly && t.args == Any[:Type, :Trivial]
    end

    function kwname(kwarg)
        kwarg isa Expr && kwarg.head === :(...) &&
            error("`@operator`: keyword varargs `kwargs...` are not supported")
        inner = kwarg isa Expr && kwarg.head === :kw ? kwarg.args[1] : kwarg
        if inner isa Expr && inner.head === :(::)
            (inner.args[1] isa Symbol) ||
                error("`@operator`: keyword arguments must have a name")
            return inner.args[1]
        end
        inner isa Symbol || error("`@operator`: keyword arguments must have a name")
        return inner
    end

    eltconstraint = parse_type_arg(posargs[1])
    for symarg in posargs[2:end]
        is_trivial_type(parse_type_arg(symarg)) ||
            error("`@operator`: symmetry arguments in the reference method must be `::Type{Trivial}`")
    end
    kwnames = map(kwname, kwtemplate)
    N = length(posargs) - 1

    loc = LineNumberNode(__source__.line, __source__.file)
    method_def(sig, body) = Expr(:function, sig, Expr(:block, loc, body))
    kwparams() = isempty(kwtemplate) ? () : (Expr(:parameters, deepcopy.(kwtemplate)...),)
    kwforward() = isempty(kwnames) ? () :
        (Expr(:parameters, (Expr(:kw, name, name) for name in kwnames)...),)
    call_expr(f, args...) = Expr(:call, f, kwforward()..., args...)
    sig_expr(f, args...) = Expr(:call, f, kwparams()..., args...)

    # symmetries are all-or-nothing: emit a no-symmetry method and an all-symmetries
    # method rather than per-argument defaults, which would also accept partial prefixes
    # (e.g. `op(sym₁)` for a two-symmetry operator) and thereby silently default the
    # remaining symmetries to `Trivial`.

    # `op(; kwargs...) = op(ComplexF64, Trivial, …; kwargs...)`
    method_none = method_def(
        sig_expr(fname),
        call_expr(fname, :ComplexF64, ntuple(Returns(:Trivial), N)...)
    )

    # `op(s₁::Type{<:Sector}, …, sₙ::Type{<:Sector}; kwargs...) =
    #      op(ComplexF64, s₁, …; kwargs...)`
    snames = [gensym(:S) for _ in 1:N]
    symsig = sig_expr(
        fname, (Expr(:(::), snames[i], :(Type{<:Sector})) for i in 1:N)...
    )
    symcall = call_expr(fname, :ComplexF64, snames...)
    method_allsyms = method_def(symsig, symcall)

    # `op(elt::<eltconstraint>; kwargs...) = op(elt, Trivial, …; kwargs...)`
    eltname = gensym(:elt)
    eltsig = sig_expr(fname, Expr(:(::), eltname, eltconstraint))
    eltcall = call_expr(fname, eltname, ntuple(Returns(:Trivial), N)...)
    method_elt = method_def(eltsig, eltcall)

    # symmetric terminal:
    # `op(elt::<c>, s₁::Type{<:Sector}, …; kwargs...) =
    #      _symmetrize_operator(op(elt, Trivial, …; kwargs...), s₁, …; kwargs...)`
    tname = gensym(:elt)
    tsyms = [gensym(:S) for _ in 1:N]
    tsig = sig_expr(
        fname, Expr(:(::), tname, eltconstraint),
        (Expr(:(::), tsyms[i], :(Type{<:Sector})) for i in 1:N)...
    )
    refcall = call_expr(fname, tname, ntuple(Returns(:Trivial), N)...)
    hookcall = call_expr(:_symmetrize_operator, refcall, tsyms...)
    method_terminal = method_def(tsig, hookcall)

    aliasdef = alias === nothing ? nothing : Expr(:const, Expr(:(=), alias, fname))

    return esc(
        quote
            $def
            Core.@__doc__ $method_none
            $method_allsyms
            $method_elt
            $method_terminal
            $loc
            $aliasdef
        end
    )
end
