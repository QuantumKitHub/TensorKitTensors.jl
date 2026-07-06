using TensorKit
using LinearAlgebra: kron
using Test
using TensorKitTensors
using TensorKitTensors: fuse_local_operators
using TensorKitTensors.SpinOperators
using TensorKitTensors.FermionOperators

@testset "desymmetrize" begin
    # spaces: graded spaces map onto ComplexSpace, preserving dimension and duality
    @test desymmetrize(spin_space(SU2Irrep; spin = 1)) == ℂ^3
    @test desymmetrize(fermion_space(U1Irrep)) == ℂ^2
    @test desymmetrize(spin_space(U1Irrep)') == (ℂ^2)'
    @test desymmetrize(ℂ^4) == ℂ^4

    # tensors over ComplexSpace are returned as-is
    X = S_x()
    @test desymmetrize(X) === X

    # desymmetrize inverts symmetrize up to the basis transformation
    U = SpinOperators.basis_transform(Z2Irrep)
    @test desymmetrize(S_x(Z2Irrep)) ≈ U * S_x() * U'

    # round trip with a trivial (identity) transformation
    t = S_exchange(SU2Irrep)
    Vd = desymmetrize(spin_space(SU2Irrep))
    @test symmetrize(desymmetrize(t), id(Vd), spin_space(SU2Irrep)) ≈ t

    # fermionic tensors desymmetrize consistently with their finer gradings
    @test desymmetrize(f_hopping(ComplexF64, Trivial)) ≈
        desymmetrize(f_hopping(ComplexF64, U1Irrep))
end

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

    # informative error naming the symmetry
    err = try
        symmetrize(S_z(), U, spin_space(Z2Irrep))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
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

    # general HomSpace form: the (codomain, domain) tuple-of-tuples entry point reproduces
    # the convenience form when the same space and transformation are used on every leg
    XX = S_x() ⊗ S_x()
    V = spin_space(Z2Irrep)
    @test symmetrize(XX, ((U, U), (U, U)), V^2 ← V^2) ≈ symmetrize(XX, (U, U), V)

    # non-square ``M ← N`` operator: permute a symmetric operator to a 3 ← 1 shape, densify,
    # and symmetrize it back onto its own (HomSpace) structure with identity transformations
    t = S_x_S_x(Z2Irrep)
    tp = permute(t, ((1, 2, 4), (3,)))
    cod_ids = ntuple(i -> id(desymmetrize(codomain(tp)[i])), numout(tp))
    dom_ids = ntuple(j -> id(desymmetrize(domain(tp)[j])), numin(tp))
    @test symmetrize(desymmetrize(tp), (cod_ids, dom_ids), space(tp)) ≈ tp
end

@testset "fuse_local_operators" begin
    # single-site: the fused operator is the Kronecker product in the fused basis.
    # TensorKit's `fuse` groups the second space as the major (slower) index, so the
    # dense representation is `kron(B, A)`.
    A = S_x()
    B = S_z()
    fused = fuse_local_operators(A, B)
    @test space(fused, 1) == fuse(space(A, 1) ⊗ space(B, 1))
    @test convert(Array, fused) ≈ kron(convert(Array, B), convert(Array, A))

    # site-wise factorization: fusing products of single-site operators must factorize,
    # which exercises the multi-site interleaving/permutation logic
    A′ = S_z()
    B′ = S_x()
    @test fuse_local_operators(A ⊗ A′, B ⊗ B′) ≈
        fuse_local_operators(A, B) ⊗ fuse_local_operators(A′, B′)

    # argument checks
    @test_throws ArgumentError fuse_local_operators(S_x(), f_num(ComplexF64, Trivial))
    @test_throws ArgumentError fuse_local_operators(S_x(), S_x_S_x())
end
