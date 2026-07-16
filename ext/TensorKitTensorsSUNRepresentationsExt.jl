module TensorKitTensorsSUNRepresentationsExt

using TensorKit
using SUNRepresentations
using LinearAlgebra: I, tr
using TensorKitTensors: symmetrize, desymmetrize
import TensorKitTensors: @operator
import TensorKitTensors.SUNOperators: sun_space, basis_transform, exchange, swap, biquadratic

_nsun(::SUNIrrep{N}) where {N} = N

# `irrep` is a weight tuple; its length fixes the SU(N) rank (SUNRepresentations requires an
# explicit `{N}` since v0.4)
_irrep(w::NTuple{N, Int}) where {N} = SUNIrrep{N}(w...)

# ---------------------------------------------------------------------------
# spaces and basis transformation
# ---------------------------------------------------------------------------

sun_space(; kwargs...) = sun_space(SUNIrrep; kwargs...)

function sun_space(::Type{<:SUNIrrep}; irrep)
    sector = _irrep(irrep)
    return Vect[typeof(sector)](sector => 1)
end

sun_space(::Type{Trivial}; irrep) = ComplexSpace(dim(_irrep(irrep)))

# SUNRepresentations' generator matrices and TensorKit's SU(N) fusion tensors both use the
# Gelfand–Tsetlin basis, so the basis transformation is the identity.
function basis_transform(symmetry::Type{<:SUNIrrep}; irrep)
    V = sun_space(symmetry; irrep)
    d = dim(V)
    return TensorMap(Matrix{Int}(I, d, d), desymmetrize(V) ← ComplexSpace(d))
end
function basis_transform(::Type{Trivial}; irrep)
    d = dim(_irrep(irrep))
    return TensorMap(Matrix{Int}(I, d, d), ComplexSpace(d) ← ComplexSpace(d))
end

function _symmetrize_operator(
        O::AbstractTensorMap, symmetry::Type{<:Sector};
        irrep = nothing, irreps = (irrep, irrep), kwargs...
    )
    V1 = sun_space(symmetry; irrep = irreps[1])
    V2 = sun_space(symmetry; irrep = irreps[2])
    U1 = basis_transform(symmetry; irrep = irreps[1])
    U2 = basis_transform(symmetry; irrep = irreps[2])
    return symmetrize(O, ((U1, U2), (U1, U2)), (V1 ⊗ V2) ← (V1 ⊗ V2))
end

# ---------------------------------------------------------------------------
# SU(N) generators (dense, in the Gelfand–Tsetlin basis)
# ---------------------------------------------------------------------------

# positive roots of su(N) in a deterministic order: (i, j) with 1 ≤ i ≤ j ≤ N-1 labels the
# root α_i + … + α_j (length increasing, so shorter roots needed by the commutators exist)
_positive_roots(N) = [(i, i + len - 1) for len in 1:(N - 1) for i in 1:(N - len)]

# the full generator basis {E_α, F_α, H_i} of sl(N) in the irrep `s`, as dense real
# matrices, in a deterministic order shared across all irreps of the same N
function _generators(::Type{T}, s::SUNIrrep{N}) where {T, N}
    simple = [Matrix{T}(collect(e)) for e in creation(s)]
    E = Dict{Tuple{Int, Int}, Matrix{T}}()
    for i in 1:(N - 1)
        E[(i, i)] = simple[i]
    end
    roots = _positive_roots(N)
    for (i, j) in roots
        i == j && continue
        E[(i, j)] = E[(i, i)] * E[(i + 1, j)] - E[(i + 1, j)] * E[(i, i)]
    end
    raising = [E[r] for r in roots]
    lowering = [Matrix(r') for r in raising]
    cartan = [simple[i] * simple[i]' - simple[i]' * simple[i] for i in 1:(N - 1)]
    return vcat(raising, lowering, cartan)
end

# inverse trace-form metric (from the fundamental) and the scalar that pins the quadratic
# Casimir onto SUNRepresentations' `casimir(2, ·)` convention. Depends only on N.
function _casimir_normalization(N)
    fund = _irrep(ntuple(i -> ifelse(i == 1, 1, 0), N))
    G = _generators(Float64, fund)
    n = length(G)
    g = [tr(G[a] * G[b]) for a in 1:n, b in 1:n]
    ginv = inv(g)
    λ = sum(ginv[a, b] * (G[a] * G[b]) for a in 1:n, b in 1:n)[1, 1]
    scale = SUNRepresentations.casimir(2, fund) / λ
    return ginv, scale
end

# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

@operator function exchange(
        elt::Type{<:Number}, ::Type{Trivial};
        irrep = nothing, irreps = (irrep, irrep)
    )
    (isnothing(irreps[1]) || isnothing(irreps[2])) &&
        throw(ArgumentError(lazy"no irrep(s) specified ($irreps)"))
    R1 = _irrep(irreps[1])
    R2 = _irrep(irreps[2])
    _nsun(R1) == _nsun(R2) ||
        throw(ArgumentError("both sites must carry irreps of the same SU(N)"))
    ginv, scale = _casimir_normalization(_nsun(R1))
    V1 = ComplexSpace(dim(R1))
    V2 = ComplexSpace(dim(R2))
    G1 = [TensorMap(g, V1 ← V1) for g in _generators(elt, R1)]
    G2 = R1 == R2 ? G1 : [TensorMap(g, V2 ← V2) for g in _generators(elt, R2)]
    n = length(G1)
    return scale * sum(ginv[a, b] * (G1[a] ⊗ G2[b]) for a in 1:n, b in 1:n)
end

@operator function biquadratic(
        elt::Type{<:Number}, ::Type{Trivial};
        irrep = nothing, irreps = (irrep, irrep)
    )
    ex = exchange(elt, Trivial; irreps)
    return ex * ex
end

@operator function swap(elt::Type{<:Number}, ::Type{Trivial}; irrep)
    d = dim(_irrep(irrep))
    V = ComplexSpace(d)
    data = zeros(elt, d, d, d, d)
    for i in 1:d, j in 1:d
        data[j, i, i, j] = one(elt)
    end
    return TensorMap(data, V ⊗ V ← V ⊗ V)
end

end
