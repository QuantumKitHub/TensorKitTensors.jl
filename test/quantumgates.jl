using TensorKit
using Test
using LinearAlgebra: I
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.QuantumGates

# gates without keywords, split by the element types they support
const REAL_GATES = (
    pauli_x, pauli_z, proj_0, proj_1, hadamard, cnot, ch, cz, swap, dcx, toffoli, fredkin,
)
const COMPLEX_GATES = (pauli_y, s_gate, t_gate, cy, cs, iswap, ecr)
# gates with a required rotation angle; `rotation_y` is the only real-valued one
const COMPLEX_ANGLE_GATES = (
    phase_shift, rotation_x, rotation_z, cphase, rotation_xx, rotation_yy, rotation_zz,
    rotation_zx,
)
# the gates that are representable with the excitation number (`U1Irrep`) and its parity
# (`Z2Irrep`) as a conserved charge
const U1_GATES = (pauli_z, proj_0, proj_1, s_gate, t_gate, cz, cs, swap, iswap, fredkin)
const U1_ANGLE_GATES = (phase_shift, rotation_z, cphase, rotation_zz)
const Z2_ANGLE_GATES = (U1_ANGLE_GATES..., rotation_xx, rotation_yy)

# a generic set of angles for the universal single-qubit gate
const U3_ANGLES = (theta = 0.73, phi = 1.17, lambda = -0.41)

# The block index of an `n`-qubit gate runs with the *first* qubit fastest,
# `k = 1 + ∑ᵢ bᵢ 2^(i-1)`, which coincides with the index convention of Qiskit's gate matrices
# when the first tensor factor is identified with qubit 0.
idx(bits::Integer...) = 1 + sum(b << (i - 1) for (i, b) in enumerate(bits))

@testset "spaces and basis transformations" begin
    @test qubit_space() == qubit_space(Trivial) == ComplexSpace(2)
    @test qubit_space(Z2Irrep) == Z2Space(0 => 1, 1 => 1)
    @test qubit_space(U1Irrep) == U1Space(0 => 1, 1 => 1)

    # all supported symmetries are diagonal in the computational basis
    for symmetry in (Trivial, Z2Irrep, U1Irrep)
        trafo = basis_transform(symmetry)
        @test trafo isa AbstractTensorMap{Int}
        @test trafo' * trafo == one(trafo)
        @test convert(Array, trafo) == I(2)
    end
end

@testset "scalar types and precision" begin
    # real scalar types stay real end-to-end
    for elt in (Float64, Float32)
        for f in REAL_GATES
            @test scalartype(f(elt)) === elt
        end
        @test scalartype(rotation_y(elt; theta = 0.7)) === elt
    end
    for elt in (ComplexF64, ComplexF32)
        for f in (REAL_GATES..., COMPLEX_GATES...)
            @test scalartype(f(elt)) === elt
        end
        for f in COMPLEX_ANGLE_GATES
            @test scalartype(f(elt; theta = 0.7)) === elt
        end
    end

    # the exactly representable gates honour integer element types
    for f in (pauli_x, pauli_z, proj_0, proj_1, cnot, cz, swap, dcx, toffoli, fredkin)
        @test scalartype(f(Int)) === Int
    end

    # symmetrizing preserves the element type
    @test scalartype(pauli_z(Float32, U1Irrep)) === Float32
    @test scalartype(swap(Float32, Z2Irrep)) === Float32
    @test scalartype(cphase(ComplexF32, U1Irrep; theta = 0.7f0)) === ComplexF32
    @test scalartype(rotation_xx(ComplexF32, Z2Irrep; theta = 0.7f0)) === ComplexF32

    # gates with entries in ℤ[i] are exact, unlike `cis(π/2)` and friends
    @test block(s_gate(), Trivial()) == ComplexF64[1 0; 0 im]
    @test block(cs(), Trivial()) == ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 im]
    @test block(cz(), Trivial()) == ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]
    @test block(proj_0(Int), Trivial()) == [1 0; 0 0]
    @test block(proj_1(Int), Trivial()) == [0 0; 0 1]

    # `signedroot` keeps the 1/√2 factors exact at any precision
    r = 1 / sqrt(big(2))
    @test abs(block(hadamard(Complex{BigFloat}), Trivial())[1, 1] - r) < eps(BigFloat)
    @test abs(block(ecr(Complex{BigFloat}), Trivial())[1, 2] - r) < eps(BigFloat)
    @test abs(block(t_gate(Complex{BigFloat}), Trivial())[2, 2] - (r + im * r)) < eps(BigFloat)

    # the element type is set by `eltype`, not by the type of the angle
    @test scalartype(rotation_x(ComplexF64; theta = big(π))) === ComplexF64
    @test block(rotation_x(Complex{BigFloat}; theta = big(π) / 2), Trivial())[1, 1] ==
        cos(big(π) / 4)

    # `u3` takes three angles, so it does not fit the loops above
    for elt in (ComplexF64, ComplexF32)
        @test scalartype(u3(elt; U3_ANGLES...)) === elt
    end

    # gates with complex matrix elements reject real element types
    for f in COMPLEX_GATES
        @test_throws ArgumentError f(Float64)
    end
    for f in COMPLEX_ANGLE_GATES
        @test_throws ArgumentError f(Float64; theta = 0.7)
    end
    @test_throws ArgumentError u3(Float64; U3_ANGLES...)
end

@testset "reference matrices" begin
    # element-wise comparison against the standard gate matrices, which pins the basis ordering
    # and, for the gates that are not symmetric under exchanging the qubits, the roles of the
    # individual qubits
    h = 1 / sqrt(2)
    theta = 0.7
    s, c = sincos(theta / 2)

    @test block(pauli_x(), Trivial()) == ComplexF64[0 1; 1 0]
    @test block(pauli_y(), Trivial()) == ComplexF64[0 -im; im 0]
    @test block(pauli_z(), Trivial()) == ComplexF64[1 0; 0 -1]
    @test block(hadamard(), Trivial()) ≈ ComplexF64[h h; h -h]
    @test block(t_gate(), Trivial()) ≈ ComplexF64[1 0; 0 (h + im * h)]
    @test block(phase_shift(; theta), Trivial()) ≈ ComplexF64[1 0; 0 cis(theta)]
    @test block(rotation_x(; theta), Trivial()) ≈ ComplexF64[c (-im * s); (-im * s) c]
    @test block(rotation_y(; theta), Trivial()) ≈ ComplexF64[c -s; s c]
    @test block(rotation_z(; theta), Trivial()) ≈ ComplexF64[cis(-theta / 2) 0; 0 cis(theta / 2)]

    # the universal single-qubit gate, and the gates it specializes to
    let (θ, φ, λ) = (U3_ANGLES.theta, U3_ANGLES.phi, U3_ANGLES.lambda)
        sθ, cθ = sincos(θ / 2)
        @test block(u3(; U3_ANGLES...), Trivial()) ≈
            ComplexF64[cθ (-cis(λ) * sθ); (cis(φ) * sθ) (cis(φ + λ) * cθ)]
        # U(θ, φ, λ) = e^{i(φ+λ)/2} Rz(φ) Ry(θ) Rz(λ)
        @test u3(; U3_ANGLES...) ≈ cis((φ + λ) / 2) *
            (rotation_z(; theta = φ) * rotation_y(; theta = θ) * rotation_z(; theta = λ))
        @test u3(; theta = 0, phi = 0, lambda = λ) ≈ phase_shift(; theta = λ)
        @test u3(; theta = π / 2, phi = 0, lambda = π) ≈ hadamard()
        @test u3(; theta = π, phi = 0, lambda = π) ≈ pauli_x()
    end

    @test block(cnot(), Trivial()) == ComplexF64[1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]
    @test block(cy(), Trivial()) == ComplexF64[1 0 0 0; 0 0 0 -im; 0 0 1 0; 0 im 0 0]
    @test block(ch(), Trivial()) ≈ ComplexF64[1 0 0 0; 0 h 0 h; 0 0 1 0; 0 h 0 -h]
    @test block(cphase(; theta), Trivial()) ≈
        ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 cis(theta)]
    @test block(swap(), Trivial()) == ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
    @test block(iswap(), Trivial()) == ComplexF64[1 0 0 0; 0 0 im 0; 0 im 0 0; 0 0 0 1]
    @test block(dcx(), Trivial()) == ComplexF64[1 0 0 0; 0 0 0 1; 0 1 0 0; 0 0 1 0]
    @test block(ecr(), Trivial()) ≈ ComplexF64[0 1 0 im; 1 0 -im 0; 0 im 0 1; -im 0 1 0] * h
    @test block(rotation_zx(; theta), Trivial()) ≈
        ComplexF64[c 0 (-im * s) 0; 0 c 0 (im * s); (-im * s) 0 c 0; 0 (im * s) 0 c]

    # the Toffoli gate flips the third qubit iff the first two are |1⟩
    ccx = Matrix{ComplexF64}(I, 8, 8)
    ccx[[idx(1, 1, 0), idx(1, 1, 1)], [idx(1, 1, 0), idx(1, 1, 1)]] = [0 1; 1 0]
    @test block(toffoli(), Trivial()) == ccx

    # the Fredkin gate swaps the last two qubits iff the first is |1⟩
    cswap = Matrix{ComplexF64}(I, 8, 8)
    cswap[[idx(1, 1, 0), idx(1, 0, 1)], [idx(1, 1, 0), idx(1, 0, 1)]] = [0 1; 1 0]
    @test block(fredkin(), Trivial()) == cswap
end

@testset "aliases" begin
    for (alias, f) in (
            X => pauli_x, Y => pauli_y, Z => pauli_z, P0 => proj_0, P1 => proj_1,
            H => hadamard, S => s_gate, T => t_gate, P => phase_shift,
            Rx => rotation_x, Ry => rotation_y, Rz => rotation_z, U => u3,
            CNOT => cnot, CX => cnot, CY => cy, CZ => cz, CP => cphase, CH => ch, CS => cs,
            SWAP => swap, ISWAP => iswap, DCX => dcx, ECR => ecr,
            Rxx => rotation_xx, Ryy => rotation_yy, Rzz => rotation_zz, Rzx => rotation_zx,
            TOFFOLI => toffoli, CCX => toffoli, FREDKIN => fredkin, CSWAP => fredkin,
        )
        @test alias === f
    end
end

@testset "controlled" begin
    q = qubit_space()

    # the controlled gates of this module are exactly `controlled` of their target
    @test controlled(pauli_x()) ≈ cnot()
    @test controlled(pauli_y()) ≈ cy()
    @test controlled(pauli_z()) ≈ cz()
    @test controlled(hadamard()) ≈ ch()
    @test controlled(s_gate()) ≈ cs()
    @test controlled(phase_shift(; theta = 0.7)) ≈ cphase(; theta = 0.7)
    @test controlled(cnot()) ≈ toffoli()
    @test controlled(swap()) ≈ fredkin()

    # controls stack
    @test controlled(controlled(pauli_x())) ≈ toffoli()

    # the element type and the symmetry are inherited from the target
    @test scalartype(controlled(pauli_x(Int))) === Int
    @test scalartype(controlled(cnot(Float32))) === Float32
    @test scalartype(controlled(rotation_x(ComplexF32; theta = 0.7f0))) === ComplexF32
    for symmetry in (Z2Irrep, U1Irrep)
        @test controlled(pauli_z(symmetry)) ≈ cz(symmetry)
        @test controlled(swap(symmetry)) ≈ fredkin(symmetry)
    end

    # gates that are not provided in controlled form, such as the controlled rotations
    crx = @testinferred controlled(rotation_x(; theta = 0.7))
    @test crx ≈ proj_0() ⊗ id(q) + proj_1() ⊗ rotation_x(; theta = 0.7)
    @test crx' * crx ≈ id(domain(crx))

    # controlling on |0⟩ instead follows from swapping the projectors
    @test proj_1() ⊗ id(q) + proj_0() ⊗ pauli_x() ≈ (pauli_x() ⊗ id(q)) * cnot() * (pauli_x() ⊗ id(q))
end

@testset "Trivial qubit gates" begin
    V = qubit_space()
    I1 = id(V)
    I2 = id(V ⊗ V)
    I3 = id(V ⊗ V ⊗ V)

    # inferrability
    x = @testinferred pauli_x()
    y = @testinferred pauli_y()
    z = @testinferred pauli_z()
    p0 = @testinferred proj_0()
    p1 = @testinferred proj_1()
    h = @testinferred hadamard()
    s = @testinferred s_gate()
    t = @testinferred t_gate()
    rx = @testinferred rotation_x(; theta = 0.7)
    ry = @testinferred rotation_y(; theta = 0.7)
    rz = @testinferred rotation_z(; theta = 0.7)
    p = @testinferred phase_shift(; theta = 0.7)
    # `@testinferred` does not accept splatted keywords, so these are spelled out
    u = @testinferred u3(; theta = 0.73, phi = 1.17, lambda = -0.41)
    cnotg = @testinferred cnot()
    cyg = @testinferred cy()
    czg = @testinferred cz()
    chg = @testinferred ch()
    csg = @testinferred cs()
    cpg = @testinferred cphase(; theta = 0.7)
    sw = @testinferred swap()
    isw = @testinferred iswap()
    d = @testinferred dcx()
    e = @testinferred ecr()
    rxx = @testinferred rotation_xx(; theta = 0.7)
    ryy = @testinferred rotation_yy(; theta = 0.7)
    rzz = @testinferred rotation_zz(; theta = 0.7)
    rzx = @testinferred rotation_zx(; theta = 0.7)
    tof = @testinferred toffoli()
    fred = @testinferred fredkin()

    # all gates are unitary
    for g in (x, y, z, h, s, t, rx, ry, rz, p, u, cnotg, cyg, czg, chg, csg, cpg, sw, isw, d, e, rxx, ryy, rzz, rzx, tof, fred)
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
    @test rotation_x(; theta = π) ≈ -im * x
    @test rotation_y(; theta = π) ≈ -im * y
    @test rotation_z(; theta = π) ≈ -im * z
    @test phase_shift(; theta = π) ≈ z
    @test phase_shift(; theta = π / 2) ≈ s
    @test phase_shift(; theta = π / 4) ≈ t

    # the angle is a required keyword
    @test_throws UndefKeywordError rotation_x()
    @test_throws UndefKeywordError phase_shift()
    @test_throws UndefKeywordError cphase()
    @test_throws UndefKeywordError rotation_zz()

    # multi-qubit relations
    @test cnotg * cnotg ≈ I2
    @test czg * czg ≈ I2
    @test sw * sw ≈ I2
    @test tof * tof ≈ I3
    @test fred * fred ≈ I3
    @test cnotg ≈ (I1 ⊗ h) * czg * (I1 ⊗ h)
    @test sw * (x ⊗ I1) * sw ≈ I1 ⊗ x
    @test cphase(; theta = π) ≈ czg
    @test isw * isw ≈ z ⊗ z
    @test chg * chg ≈ I2           # controlled-H is an involution
    @test csg * csg ≈ czg          # controlled-S squares to CZ
    @test d * d * d ≈ I2           # DCX has order 3
    @test e * e ≈ I2               # ECR is an involution
    @test e' ≈ e                   # ECR is Hermitian

    # two-qubit rotations
    @test rotation_xx(; theta = π) ≈ -im * (x ⊗ x)
    @test rotation_yy(; theta = π) ≈ -im * (y ⊗ y)
    @test rotation_zz(; theta = π) ≈ -im * (z ⊗ z)
    @test rotation_zx(; theta = π) ≈ -im * (z ⊗ x)
end

@testset "U1-symmetric qubit gates" begin
    trafo = basis_transform(U1Irrep)

    # inferrability
    @test (@testinferred pauli_z(U1Irrep)) isa AbstractTensorMap
    @test (@testinferred proj_0(U1Irrep)) isa AbstractTensorMap
    @test (@testinferred cz(U1Irrep)) isa AbstractTensorMap
    @test (@testinferred swap(U1Irrep)) isa AbstractTensorMap
    @test (@testinferred iswap(U1Irrep)) isa AbstractTensorMap
    @test (@testinferred fredkin(U1Irrep)) isa AbstractTensorMap
    @test (@testinferred phase_shift(U1Irrep; theta = 0.7)) isa AbstractTensorMap
    @test (@testinferred rotation_zz(U1Irrep; theta = 0.7)) isa AbstractTensorMap

    # the symmetric gates match their trivial versions element-wise
    for f in U1_GATES
        test_operator_dense(f(U1Irrep), f(Trivial), trafo)
    end
    for f in U1_ANGLE_GATES
        test_operator_dense(f(U1Irrep; theta = 0.7), f(Trivial; theta = 0.7), trafo)
    end
end

@testset "Z2-symmetric qubit gates" begin
    trafo = basis_transform(Z2Irrep)

    # inferrability
    @test (@testinferred pauli_z(Z2Irrep)) isa AbstractTensorMap
    @test (@testinferred swap(Z2Irrep)) isa AbstractTensorMap
    @test (@testinferred fredkin(Z2Irrep)) isa AbstractTensorMap
    @test (@testinferred rotation_xx(Z2Irrep; theta = 0.7)) isa AbstractTensorMap

    # the symmetric gates match their trivial versions element-wise
    for f in U1_GATES
        test_operator_dense(f(Z2Irrep), f(Trivial), trafo)
    end
    for f in Z2_ANGLE_GATES
        test_operator_dense(f(Z2Irrep; theta = 0.7), f(Trivial; theta = 0.7), trafo)
    end
end

@testset "unsupported symmetries" begin
    # gates that flip a single qubit conserve neither the excitation number nor its parity
    for f in (pauli_x, pauli_y, hadamard, cnot, cy, ch, dcx, ecr, toffoli)
        @test_throws ArgumentError f(U1Irrep)
        @test_throws ArgumentError f(Z2Irrep)
    end
    for f in (rotation_x, rotation_y, rotation_zx)
        @test_throws ArgumentError f(U1Irrep; theta = 0.5)
        @test_throws ArgumentError f(Z2Irrep; theta = 0.5)
    end

    # the universal single-qubit gate mixes |0⟩ and |1⟩ for a generic angle
    @test_throws ArgumentError u3(U1Irrep; U3_ANGLES...)
    @test_throws ArgumentError u3(Z2Irrep; U3_ANGLES...)

    # the XX and YY couplings flip both qubits, conserving parity but not the excitation number
    for f in (rotation_xx, rotation_yy)
        @test_throws ArgumentError f(U1Irrep; theta = 0.5)
    end

    # SU2 is not implemented for any gate
    @test_throws ArgumentError qubit_space(SU2Irrep)
    for f in (pauli_z, proj_0, swap, cz)
        @test_throws ArgumentError f(SU2Irrep)
    end
end
