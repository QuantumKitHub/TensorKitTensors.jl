using TensorKit
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.QuantumGates

@testset "Trivial qubit gates" begin
    V = qubit_space()
    I1 = id(V)
    I2 = id(V ⊗ V)
    I3 = id(V ⊗ V ⊗ V)

    # inferrability
    x = @inferred pauli_x()
    y = @inferred pauli_y()
    z = @inferred pauli_z()
    p0 = @inferred proj_0()
    p1 = @inferred proj_1()
    h = @inferred hadamard()
    s = @inferred s_gate()
    t = @inferred t_gate()
    rx = @inferred rotation_x(; θ = 0.7)
    ry = @inferred rotation_y(; θ = 0.7)
    rz = @inferred rotation_z(; θ = 0.7)
    p = @inferred phase_shift(; θ = 0.7)
    cnotg = @inferred cnot()
    cyg = @inferred cy()
    czg = @inferred cz()
    chg = @inferred ch()
    csg = @inferred cs()
    cpg = @inferred cphase(; θ = 0.7)
    sw = @inferred swap()
    isw = @inferred iswap()
    d = @inferred dcx()
    e = @inferred ecr()
    tof = @inferred toffoli()
    fred = @inferred fredkin()

    # all gates are unitary
    for g in (x, y, z, h, s, t, rx, ry, rz, p, cnotg, cyg, czg, chg, csg, cpg, sw, isw, d, e, tof, fred)
        @test g' * g ≈ id(domain(g))
    end

    # Pauli algebra
    @test x * x ≈ I1
    @test y * y ≈ I1
    @test z * z ≈ I1
    @test x * y ≈ im * z
    @test y * z ≈ im * x
    @test z * x ≈ im * y

    # projectors
    @test p0 + p1 ≈ I1
    @test p0 - p1 ≈ z
    @test p0 * p0 ≈ p0
    @test p1 * p1 ≈ p1
    @test p0 * p1 ≈ zero(p0)

    # Clifford relations
    @test h * h ≈ I1
    @test h * x * h ≈ z
    @test h * z * h ≈ x
    @test s * s ≈ z
    @test t * t ≈ s

    # rotations and phase gates
    @test rotation_x(; θ = π) ≈ -im * x
    @test rotation_y(; θ = π) ≈ -im * y
    @test rotation_z(; θ = π) ≈ -im * z
    @test phase_shift(; θ = π) ≈ z
    @test phase_shift(; θ = π / 2) ≈ s
    @test phase_shift(; θ = π / 4) ≈ t

    # the ASCII `theta` keyword is an alias for `θ`
    @test rotation_x(; theta = 0.7) ≈ rotation_x(; θ = 0.7)
    @test rotation_z(; theta = π) ≈ rotation_z(; θ = π)
    @test phase_shift(; theta = 0.7) ≈ phase_shift(; θ = 0.7)
    @test cphase(; theta = 0.7) ≈ cphase(; θ = 0.7)

    # missing angle is an error
    @test_throws ArgumentError rotation_x()
    @test_throws ArgumentError phase_shift()
    @test_throws ArgumentError cphase()

    # multi-qubit relations
    @test cnotg * cnotg ≈ I2
    @test czg * czg ≈ I2
    @test sw * sw ≈ I2
    @test tof * tof ≈ I3
    @test fred * fred ≈ I3
    @test cnotg ≈ (I1 ⊗ h) * czg * (I1 ⊗ h)
    @test sw * (x ⊗ I1) * sw ≈ I1 ⊗ x
    @test cphase(; θ = π) ≈ czg
    @test isw * isw ≈ z ⊗ z
    @test chg * chg ≈ I2           # controlled-H is an involution
    @test csg * csg ≈ czg          # controlled-S squares to CZ
    @test d * d * d ≈ I2            # DCX has order 3
    @test e * e ≈ I2               # ECR is an involution
    @test e' ≈ e                   # ECR is Hermitian
    @test e ≈ (I1 ⊗ x - x ⊗ y) / sqrt(2)
end

@testset "U1-symmetric qubit gates" begin
    # inferrability
    @inferred pauli_z(U1Irrep)
    @inferred proj_0(U1Irrep)
    @inferred cz(U1Irrep)
    @inferred swap(U1Irrep)
    @inferred iswap(U1Irrep)
    @inferred fredkin(U1Irrep)
    @inferred phase_shift(U1Irrep; θ = 0.7)

    # the symmetric gates match their trivial versions
    for f in (pauli_z, proj_0, proj_1, s_gate, t_gate, cz, cs, swap, iswap, fredkin)
        test_operator(f(U1Irrep), f(Trivial))
    end
    for f in (phase_shift, rotation_z, cphase)
        test_operator(f(U1Irrep; θ = 0.7), f(Trivial; θ = 0.7))
    end
end

@testset "unsupported symmetries" begin
    # gates that break excitation-number conservation reject U1Irrep
    for f in (pauli_x, pauli_y, hadamard, cnot, cy, ch, dcx, ecr, toffoli)
        @test_throws ArgumentError f(U1Irrep)
    end
    @test_throws ArgumentError rotation_x(U1Irrep; θ = 0.5)
    @test_throws ArgumentError rotation_y(U1Irrep; θ = 0.5)

    # Z2 and SU2 are not implemented for any gate
    for symm in (Z2Irrep, SU2Irrep)
        @test_throws ArgumentError qubit_space(symm)
        @test_throws ArgumentError pauli_z(symm)
        @test_throws ArgumentError proj_0(symm)
        @test_throws ArgumentError swap(symm)
        @test_throws ArgumentError cz(symm)
    end
end
