module TensorKitTensorsSUNRepresentationsExt

using TensorKitTensors.SUNOperators
using TensorKitTensors: desymmetrize
using SUNRepresentations
using TensorKit
using LinearAlgebra: diagind

import TensorKitTensors.SUNOperators: sun_space, exchange, swap, twosite_casimir

# ---------------------------------------------------------------------------
# sun_space
# ---------------------------------------------------------------------------

sun_space(; kwargs...) = sun_space(SUNIrrep; kwargs...)

function sun_space(I::Type{<:SUNIrrep}; irrep)
    sector = I(irrep)
    return Vect[typeof(sector)](sector => 1)
end

function sun_space(::Type{Trivial}; irrep)
    sector = SUNIrrep(irrep)
    return ComplexSpace(dim(sector))
end

# ---------------------------------------------------------------------------
# twosite_casimir
# ---------------------------------------------------------------------------

twosite_casimir(; kwargs...) = twosite_casimir(Float64, SUNIrrep; kwargs...)
twosite_casimir(elt::Type{<:Number}; kwargs...) = twosite_casimir(elt, SUNIrrep; kwargs...)
twosite_casimir(symmetry::Type; kwargs...) = twosite_casimir(Float64, symmetry; kwargs...)

function twosite_casimir(
        elt::Type{<:Number}, symmetry::Type{<:SUNIrrep};
        k::Int = 2, irrep = nothing, irreps::NTuple{2, Any} = (irrep, irrep)
    )
    (isnothing(irreps[1]) | isnothing(irreps[2])) &&
        throw(ArgumentError(lazy"invalid dynkin labels specified ($irreps)"))
    V1 = sun_space(symmetry; irrep = irreps[1])
    V2 = sun_space(symmetry; irrep = irreps[2])
    T = zeros(elt, V1 ⊗ V2 ← V1 ⊗ V2)
    for (c, b) in blocks(T)
        val = casimir(k, c)
        @inbounds for i in diagind(b)
            b[i] = val
        end
    end
    return T
end

twosite_casimir(elt::Type{<:Number}, ::Type{Trivial}; kwargs...) =
    desymmetrize(twosite_casimir(elt, SUNIrrep; kwargs...))

# ---------------------------------------------------------------------------
# exchange
# ---------------------------------------------------------------------------

exchange(; kwargs...) = exchange(Float64, SUNIrrep; kwargs...)
exchange(elt::Type{<:Number}; kwargs...) = exchange(elt, SUNIrrep; kwargs...)
exchange(symmetry::Type; kwargs...) = exchange(Float64, symmetry; kwargs...)

function exchange(
        elt::Type{<:Number}, ::Type{SUNIrrep};
        irrep = nothing, irreps::NTuple{2, Any} = (irrep, irrep)
    )
    (isnothing(irreps[1]) | isnothing(irreps[2])) &&
        throw(ArgumentError(lazy"invalid dynkin labels specified ($irreps)"))
    V1 = sun_space(SUNIrrep; irrep = irreps[1])
    V2 = sun_space(SUNIrrep; irrep = irreps[2])
    T = zeros(elt, V1 ⊗ V2 ← V1 ⊗ V2)
    c²₁ = casimir(2, only(sectors(V1)))
    c²₂ = casimir(2, only(sectors(V2)))
    for (c, b) in blocks(T)
        c² = casimir(2, c)
        val = (c² - c²₁ - c²₂) / 2
        @inbounds for i in diagind(b)
            b[i] = val
        end
    end
    return T
end

exchange(elt::Type{<:Number}, ::Type{Trivial}; kwargs...) =
    desymmetrize(exchange(elt, SUNIrrep; kwargs...))

# ---------------------------------------------------------------------------
# swap (symmetric only — permutation only defined for equal spaces)
# ---------------------------------------------------------------------------

swap(; kwargs...) = swap(Float64, SUNIrrep; kwargs...)
swap(elt::Type{<:Number}; kwargs...) = swap(elt, SUNIrrep; kwargs...)
swap(symmetry::Type; kwargs...) = swap(Float64, symmetry; kwargs...)

function swap(elt::Type{<:Number}, ::Type{SUNIrrep}; irrep)
    ex = exchange(elt, SUNIrrep; irrep)
    N = SUNIrrep(irrep...).N
    return add(ex, id(domain(ex)), 1 // N, 2)
end

swap(elt::Type{<:Number}, ::Type{Trivial}; kwargs...) =
    desymmetrize(swap(elt, SUNIrrep; kwargs...))

end
