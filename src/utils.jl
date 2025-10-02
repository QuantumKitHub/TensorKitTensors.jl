# ---- copied from MPSKit, PEPSKit ----

_totuple(t) = t isa Tuple ? t : Tuple(t)

"""
    tensorexpr(name, ind_out, [ind_in])

Generates expressions for use within `@tensor` environments
of the form `name[ind_out...; ind_in]`.
"""
tensorexpr(name, inds) = Expr(:ref, name, _totuple(inds)...)
function tensorexpr(name, indout, indin)
    return Expr(
        :typed_vcat, name, Expr(:row, _totuple(indout)...), Expr(:row, _totuple(indin)...)
    )
end

"""
    twistdual(t::AbstractTensorMap, i)
    twistdual!(t::AbstractTensorMap, i)

Twist the i-th leg of a tensor `t` if it represents a dual space.
"""
function twistdual!(t::AbstractTensorMap, i::Int)
    isdual(space(t, i)) || return t
    return twist!(t, i)
end
function twistdual!(t::AbstractTensorMap, is)
    is′ = filter(i -> isdual(space(t, i)), is)
    return twist!(t, is′)
end
twistdual(t::AbstractTensorMap, is) = twistdual!(copy(t), is)


@generated function _fuse_isomorphisms(
        op::AbstractTensorMap{<:Any, S, N, N}, fs::Vector{<:AbstractTensorMap{<:Any, S, 1, 2}}
    ) where {S, N}
    op_out_e = tensorexpr(:op_out, -(1:N), -((1:N) .+ N))
    op_e = tensorexpr(:op, 1:3:(3 * N), 2:3:(3 * N))
    f_es = map(1:N) do i
        j = 3 * (i - 1) + 1
        return tensorexpr(:(fs[$i]), -i, (j, j + 2))
    end
    f_dag_es = map(1:N) do i
        j = 3 * (i - 1) + 1
        return tensorexpr(:(twistdual(fs[$i]', 1:2)), (j + 1, j + 2), -(N + i))
    end
    multiplication_ex = Expr(
        :call, :*, op_e, f_es..., f_dag_es...
    )
    return macroexpand(@__MODULE__, :(return @tensor $op_out_e := $multiplication_ex))
end

"""
    _fuse_ids(op::AbstractTensorMap{T, S, N, N}, [Ps::NTuple{N, S}]) where {T, S, N}

Fuse identities on auxiliary physical spaces `Ps` into a given operator `op`.
When `Ps` is not specified, it defaults to the domain spaces of `op`.
"""
function _fuse_ids(op::AbstractTensorMap{T, S, N, N}, Ps::NTuple{N, S}) where {T, S, N}
    # make isomorphisms
    fs = map(1:N) do i
        return isomorphism(fuse(space(op, i), Ps[i]), space(op, i) ⊗ Ps[i])
    end
    # and fuse them into the operator
    return _fuse_isomorphisms(op, fs)
end

# ----
