using TensorKit
using LinearAlgebra: I
using Test
using TensorKitTensors
using TensorKitTensors.SpinOperators
using TensorKitTensors.FermionOperators

@testset "symmetrize" begin
    # single-site round trip: transverse-field Ising term with Z2 symmetry
    X = S_x()
    U = SpinOperators.basis_transform(Z2Irrep)
    X_z2 = symmetrize(X, U, spin_space(Z2Irrep))
    @test block(X_z2, Z2Irrep(0)) ≈ fill(1 / 2, 1, 1)
    @test block(X_z2, Z2Irrep(1)) ≈ fill(-1 / 2, 1, 1)
    @test X_z2 ≈ S_x(Z2Irrep)

    # per-site tuple form
    XX = S_x() ⊗ S_x()
    XX_z2 = symmetrize(XX, (U, U), spin_space(Z2Irrep))
    @test XX_z2 ≈ S_x_S_x(Z2Irrep)

    # informative error naming the operator
    err = try
        symmetrize(S_z(), U, spin_space(Z2Irrep); name = "S_z")
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("S_z", err.msg)
    @test occursin("Z2Irrep", err.msg)

    # tol kwarg: a slightly perturbed operator projects with loose tolerance
    X_pert = deepcopy(X)
    X_pert[1, 1] += 1.0e-6
    @test_throws ArgumentError symmetrize(X_pert, U, spin_space(Z2Irrep))
    X_loose = symmetrize(X_pert, U, spin_space(Z2Irrep); tol = 1.0e-3)
    @test X_loose ≈ X_z2 atol = 1.0e-5

    # fermionic input: fZ2-graded operator projected into the finer U1 grading
    Uf = FermionOperators.basis_transform(U1Irrep)
    f_u1 = symmetrize(f_hopping(ComplexF64, Trivial), Uf, fermion_space(U1Irrep))
    @test f_u1 ≈ f_hopping(ComplexF64, U1Irrep)

    # argument checks
    @test_throws ArgumentError symmetrize(XX, (U,), spin_space(Z2Irrep))

    # the default tol is floored at Float64 resolution, such that wide scalar types work
    # with non-abelian symmetries (whose fusion-tensor data is Float64)
    SS_big = symmetrize(
        S_exchange(Complex{BigFloat}), SpinOperators.basis_transform(SU2Irrep),
        spin_space(SU2Irrep)
    )
    @test scalartype(SS_big) === Complex{BigFloat}
end
