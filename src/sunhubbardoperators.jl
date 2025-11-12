"""
    module SUNHubbardOperators

Module for the ``SU(N)`` generalizations of the [`HubbardOperators`](@ref) operators.

!!! note
    This module requires `using SUNRepresentations` in order to define the `SUNIrrep`-symmetric tensors.
"""
module SUNHubbardOperators

using TensorKit

export hubbard_space
"""
    hubbard_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; N::Integer = 3)

Return the local hilbert space for a Hubbard-type model with the given particle and spin symmetries. 
The basis is spanned by all possible fermionic color-like combinations of particles that satisfy the Pauli exclusion principle.

For example, for ``SU(3)`` (`N = 3`), we have three colors (`r`, `g` and `b`), leading to the following eight possible states:

```math
|0⟩, |r⟩, |g⟩, |b⟩, |rg⟩, |rb⟩, |gb⟩, |rgb⟩
```

Additionally, we can implement `Trivial` or `U1Irrep` symmetry for the particle number conservation.
"""
function hubbard_space(::Type{PS} = Trivial, ::Type{SS} = Trivial; N::Integer = 3) where {PS <: Sector, SS <: Sector}
    if SS === Trivial
        if PS === Trivial
            N == 2 && return Vect[FermionParity](0 => 2, 1 => 2)
            N == 3 && return Vect[FermionParity](0 => 4, 1 => 4)
        elseif PS === U1Irrep
            N == 2 && return Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 2, (0, 2) => 1)
            N == 3 && return Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 3, (0, 2) => 3, (1, 3) => 1)
        end
    end

    error(lazy"combination of particle symmetry ($PS) and spin symmetry ($SS) not supported or not implemented for N = $N")
end

end
