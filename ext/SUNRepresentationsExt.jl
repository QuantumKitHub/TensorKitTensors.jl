module SUNRepresentationsExt

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


end
