using TensorKit
using LinearAlgebra: tr
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.HubbardOperators
using StableRNGs

@testset "Symmetric operators with symmetries $(particle_symmetry) and $(spin_symmetry)" for particle_symmetry = [Trivial, U1Irrep, SU2Irrep], spin_symmetry = [Trivial, U1Irrep, SU2Irrep]
    space = hubbard_space(particle_symmetry, spin_symmetry)

    O = e_plusmin(ComplexF64, particle_symmetry, spin_symmetry)
    O_triv = e_plusmin(ComplexF64, Trivial, Trivial)
    test_operator(O, O_triv)

    O = e_number(ComplexF64, particle_symmetry, spin_symmetry)
    O_triv = e_number(ComplexF64, Trivial, Trivial)
    test_operator(O, O_triv)

    O = e_number_updown(ComplexF64, particle_symmetry, spin_symmetry)
    O_triv = e_number_updown(ComplexF64, Trivial, Trivial)
    test_operator(O, O_triv)
end