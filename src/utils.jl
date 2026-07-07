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
    @operator [alias] op(::Type{<:Number}[=default], ::Type{<:Sector}[=default], …; kwargs...)

Generate the public default-argument interface for an operator whose non-symmetric reference
implementation `op(elt, Trivial, …; kwargs...)` (with a concrete `::Type{Trivial}` for each
symmetry slot) is written separately.

The signature is a *template* describing the public interface: an element type as the first
positional argument, followed by one or more symmetry arguments, and optionally some keyword
arguments. Each argument's type annotation constrains the corresponding generated method, and
each `= default` supplies the value filled in when the argument is omitted (the element type
defaults to `ComplexF64` and symmetries to `Trivial`). `@operator` emits the boilerplate so
that all of the following resolve:

- `op()`
- `op(eltype)`
- `op(symmetry₁, …, symmetryₙ)` (any prefix; remaining symmetries take their defaults)
- `op(eltype, symmetry₁, …, symmetryₙ)`

The last of these, the *symmetric terminal*, delegates to the module-local hook
`_symmetrize_operator(op(elt, Trivial, …; kwargs...), symmetry₁, …; kwargs...)`, which each
operator module defines once (typically wrapping [`symmetrize`](@ref) with the module's
`basis_transform` and local space). Keyword arguments (`spin`, `cutoff`, …) are forwarded
generically to both the reference call and the hook; the `kwargs` in the template are only for
documentation.

If an `alias` symbol is supplied, a `const alias = op` binding is emitted as well (e.g. the
Unicode name `Sᶻ` for `S_z`).

The first generated method is wrapped in `Core.@__doc__`, so a docstring written directly
above the `@operator` line is attached to the operator function. Write the reference
method(s) as ordinary definitions, either above or below the `@operator` line:

```julia
\"\"\"
    S_z([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin-z operator.
\"\"\"
@operator Sᶻ S_z(::Type{<:Number}, ::Type{<:Sector}; spin)
function S_z(elt::Type{<:Number}, ::Type{Trivial}; spin = 1 // 2)
    ...
end
```
"""
macro operator(args...)
    (1 <= length(args) <= 2) ||
        error("`@operator` expects `[alias] op(signature)`")
    if length(args) == 2
        alias, sig = args[1], args[2]
        (alias isa Symbol) || error("`@operator`: the alias must be a symbol")
    else
        alias, sig = nothing, args[1]
    end

    (sig isa Expr && sig.head === :call) ||
        error("`@operator` expects a signature template `op(::Type{...}, …)`")
    fname = sig.args[1]
    (fname isa Symbol) || error("`@operator`: the operator name must be a symbol")
    posargs = filter(a -> !(a isa Expr && a.head === :parameters), sig.args[2:end])
    length(posargs) >= 2 ||
        error("`@operator` expects an element type plus at least one symmetry argument")

    # parse an argument into (type constraint, default-or-nothing)
    function parse_arg(a)
        inner, default = (a isa Expr && a.head === :kw) ? (a.args[1], a.args[2]) : (a, nothing)
        (inner isa Expr && inner.head === :(::)) ||
            error("`@operator`: each argument must be `::Type{...}` (optionally `= default`)")
        return inner.args[end], default
    end

    # set default values if not given: `elt = ComplexF64` and `symmetry = Trivial`
    eltconstraint, eltdefault = parse_arg(posargs[1])
    eltdefault === nothing && (eltdefault = :ComplexF64)
    syms = map(parse_arg, posargs[2:end])
    symconstraints = first.(syms)
    symdefaults = map(s -> s[2] === nothing ? :Trivial : s[2], syms)
    N = length(syms)

    kw = gensym(:kwargs)
    kwparam = Expr(:parameters, Expr(:(...), kw))

    # `op(s₁::<c₁> = <d₁>, …; kwargs...) = op(<eltdefault>, s₁, …; kwargs...)`
    snames = [gensym(:S) for _ in 1:N]
    symsig = Expr(
        :call, fname, kwparam,
        (Expr(:kw, Expr(:(::), snames[i], symconstraints[i]), symdefaults[i]) for i in 1:N)...
    )
    symcall = Expr(:call, fname, kwparam, eltdefault, snames...)
    method_sym = Expr(:(=), symsig, symcall)

    # `op(elt::<eltconstraint>; kwargs...) = op(elt, <d₁>, …; kwargs...)`
    eltname = gensym(:elt)
    eltsig = Expr(:call, fname, kwparam, Expr(:(::), eltname, eltconstraint))
    eltcall = Expr(:call, fname, kwparam, eltname, symdefaults...)
    method_elt = Expr(:(=), eltsig, eltcall)

    # symmetric terminal:
    # `op(elt::<c>, s₁::Type{<:Sector}, …; kwargs...) =
    #      _symmetrize_operator(op(elt, Trivial, …; kwargs...), s₁, …; kwargs...)`
    tname = gensym(:elt)
    tsyms = [gensym(:S) for _ in 1:N]
    tsig = Expr(
        :call, fname, kwparam, Expr(:(::), tname, eltconstraint),
        (Expr(:(::), tsyms[i], symconstraints[i]) for i in 1:N)...
    )
    refcall = Expr(:call, fname, kwparam, tname, ntuple(Returns(:Trivial), N)...)
    hookcall = Expr(:call, :_symmetrize_operator, kwparam, refcall, tsyms...)
    method_terminal = Expr(:(=), tsig, hookcall)

    aliasdef = alias === nothing ? nothing : Expr(:const, Expr(:(=), alias, fname))

    return esc(
        quote
            Core.@__doc__ $method_sym
            $method_elt
            $method_terminal
            $aliasdef
        end
    )
end
