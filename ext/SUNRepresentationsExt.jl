module SUNRepresentationsExt

using TensorKit
using SUNRepresentations
using TensorKitTensors.SUNHubbardOperators

function SUNHubbardOperators.hubbard_space(::Type{Trivial}, ::Type{SUNIrrep{3}}; N::Integer = 3)
    @assert N == 3 "incompatible values of N"
    return Vect[FermionParity ⊠ SUNIrrep{3}](
        (0, (0, 0, 0)) => 1,    # |0⟩
        (1, (1, 0, 0)) => 1,    # |r⟩, |g⟩, |b⟩
        (0, (1, 1, 0)) => 1,    # |rg⟩, |rb⟩, |gb⟩
        (1, (0, 0, 0)) => 1     # |rgb⟩
    )
end
function SUNHubbardOperators.hubbard_space(::Type{U1Irrep}, ::Type{SUNIrrep{3}}; N::Integer = 3)
    @assert N == 3 "incompatible values of N"
    return Vect[FermionParity ⊠ U1Irrep ⊠ SUNIrrep{3}](
        (0, 0, (0, 0, 0)) => 1,    # |0⟩
        (1, 1, (1, 0, 0)) => 1,    # |r⟩, |g⟩, |b⟩
        (0, 2, (1, 1, 0)) => 1,    # |rg⟩, |rb⟩, |gb⟩
        (1, 3, (0, 0, 0)) => 1     # |rgb⟩
    )
end

# Single site operators
# ---------------------
function SUNHubbardOperators.e_num(::Type{T}, ::Type{Trivial}, ::Type{SUNIrrep{3}}; kwargs...) where {T <: Number}
    t = SUNHubbardOperators.single_site_operator(T, Trivial, SU3Irrep; kwargs...)
    I = sectortype(t)
    for (i, c) in enumerate(((0, (0, 0, 0)), (1, (1, 0, 0)), (0, (1, 1, 0)), (1, (0, 0, 0))))
        block(t, I(c)) .= i - 1
    end
    return t
end
function SUNHubbardOperators.e_num(::Type{T}, ::Type{U1Irrep}, ::Type{SUNIrrep{3}}; kwargs...) where {T <: Number}
    t = SUNHubbardOperators.single_site_operator(T, U1Irrep, SU3Irrep; kwargs...)
    I = sectortype(t)
    for c in ((0, (0, 0, 0)), (1, (1, 0, 0)), (2, (1, 1, 0)), (3, (0, 0, 0)))
        block(t, I(c)) .= c[1]
    end
    return t
end


# Two site operators
# ------------------
# TODO: it might be possible to create "analytic" expressions for this which involve 2 Bmoves
# and an Fmove, but this should give the same results

function SUNHubbardOperators.e_plus_e_min(::Type{T}, ::Type{U1Irrep}, ::Type{SUNIrrep{3}}; kwargs...) where {T <: Number}
    V = SUNHubbardOperators.hubbard_space(U1Irrep, SU3Irrep; kwargs...)
    A = typeof(V)((1, 1, (1, 0, 0)) => 1)

    L = zeros(V ← V ⊗ A)
    for (f₁, f₂) in fusiontrees(L)
        n₁ = f₂.uncoupled[1][1].isodd
        L[f₁, f₂] .= (-1)^n₁
    end
    R = ones(A ⊗ V ← V)

    return @planar t[-1 -2; -3 -4] := L[-1; -3 1] * R[1 -2; -4]
end

function SUNHubbardOperators.e_min_e_plus(::Type{T}, ::Type{U1Irrep}, ::Type{SUNIrrep{3}}; kwargs...) where {T <: Number}
    V = SUNHubbardOperators.hubbard_space(U1Irrep, SU3Irrep; kwargs...)
    A = typeof(V)((1, 1, (1, 0, 0)) => 1)

    L = zeros(V ⊗ A ← V)
    for (f₁, f₂) in fusiontrees(L)
        n₁ = f₂.uncoupled[1][1].isodd
        L[f₁, f₂] .= (-1)^n₁
    end
    R = ones(V ← A ⊗ V)

    return @planar t[-1 -2; -3 -4] := L[-1 1; -3] * R[-2; 1 -4]
end

end
