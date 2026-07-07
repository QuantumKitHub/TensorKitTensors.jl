using TensorKit
using LinearAlgebra: eigvals
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors: desymmetrize
using TensorKitTensors.HubbardOperators
using StableRNGs

particle_syms = (Trivial, U1Irrep, SU2Irrep)
spin_syms = (Trivial, U1Irrep, SU2Irrep)

# operator availability, as determined by the symmetries each operator breaks:
# - u_num, d_num, S_z and the u/d hopping pairs break SU2 spin symmetry
# - e_num, ud_num, h_num and e_plus_e_min/e_min_e_plus break SU2 particle symmetry
# - the pair (charge non-conserving) operators break U1 and SU2 particle symmetry
# - half_ud_num, e_hopping, S_exchange are compatible with all symmetries
has_u_num(P, S) = P !== SU2Irrep && S !== SU2Irrep
has_e_num(P, S) = P !== SU2Irrep
has_spin_ops(P, S) = S === Trivial
has_S_z(P, S) = S !== SU2Irrep
has_e_pm(P, S) = P !== SU2Irrep
has_pair(P, S) = P === Trivial && S !== SU2Irrep
has_singlet(P, S) = P === Trivial
has_triplet(P, S) = P === Trivial && S === Trivial

@testset "basis transformations" begin
    for (P, S) in Iterators.product(particle_syms, spin_syms)
        U = basis_transform(P, S)
        @test U isa AbstractTensorMap
        @test scalartype(U) === Int # exact entries promote without precision loss
        @test U' * U == one(U)
        @test U * U' == one(U)
    end
    @test basis_transform(Trivial, Trivial) == one(basis_transform(Trivial, Trivial))

    # real and wide scalar types are preserved; the staggered gauge only forces complex
    # entries where the operator is genuinely complex
    @test scalartype(u_num(Float64, U1Irrep, U1Irrep)) === Float64
    @test scalartype(half_ud_num(Float64, SU2Irrep, SU2Irrep)) === Float64
    @test scalartype(e_hopping(Complex{BigFloat}, SU2Irrep, SU2Irrep)) === Complex{BigFloat}
    N_big = u_num(BigFloat, U1Irrep, U1Irrep)
    @test all(((c, b),) -> all(isinteger, b), blocks(N_big))
end

@testset "Compare symmetric with trivial tensors" begin
    for (particle_symmetry, spin_symmetry) in Iterators.product(particle_syms, spin_syms)
        space = @inferred hubbard_space(particle_symmetry, spin_symmetry)
        @test dim(space) == 4

        # element-wise comparison in the dense basis catches transposes and gauge errors
        # that the spectral `test_operator` is blind to
        U = basis_transform(particle_symmetry, spin_symmetry)

        # compatible with all symmetry combinations
        for f in (half_ud_num, S_exchange)
            O = f(ComplexF64, particle_symmetry, spin_symmetry)
            O_triv = f(ComplexF64, Trivial, Trivial)
            test_operator_dense(O, O_triv, U)
        end

        # e_hopping uses the staggered η-pairing gauge c_{j,σ} → iʲ c_{j,σ} under SU2 particle
        # symmetry (Yang & Zhang, Mod. Phys. Lett. B 4, 759 (1990), the SO₄ symmetry of the
        # Hubbard model), so site k carries the extra factor G^(k-1) with G = iⁿ = diag(1, -1,
        # i, i) in the (|0⟩, |↑↓⟩, |↑⟩, |↓⟩) basis. Making the site-dependent transform explicit
        # lets the element-wise comparison apply here too, rather than falling back to the spectrum.
        if particle_symmetry === SU2Irrep
            Vref = desymmetrize(hubbard_space(Trivial, Trivial))
            G = TensorMap(Complex{Int}[1 0 0 0; 0 -1 0 0; 0 0 im 0; 0 0 0 im], Vref ← Vref)
            test_operator_dense(
                e_hopping(ComplexF64, particle_symmetry, spin_symmetry),
                e_hopping(ComplexF64, Trivial, Trivial), (U, U * G)
            )
        else
            test_operator_dense(
                e_hopping(ComplexF64, particle_symmetry, spin_symmetry),
                e_hopping(ComplexF64, Trivial, Trivial), U
            )
        end

        for (available, fs) in (
                (has_e_num(particle_symmetry, spin_symmetry), (e_num, ud_num)),
                (has_u_num(particle_symmetry, spin_symmetry), (u_num, d_num)),
                (has_e_pm(particle_symmetry, spin_symmetry), (e_plus_e_min,)),
                (has_singlet(particle_symmetry, spin_symmetry), (singlet_plus,)),
                (has_S_z(particle_symmetry, spin_symmetry), (S_plus_S_min, S_min_S_plus)),
            )
            for f in fs
                if available
                    O = f(ComplexF64, particle_symmetry, spin_symmetry)
                    O_triv = f(ComplexF64, Trivial, Trivial)
                    test_operator_dense(O, O_triv, U)
                else
                    @test_throws ArgumentError f(
                        ComplexF64, particle_symmetry, spin_symmetry
                    )
                end
            end
        end
    end
end

@testset "regression values" begin
    # staggered-gauge e_hopping with SU2 x SU2 symmetry
    t = e_hopping(ComplexF64, SU2Irrep, SU2Irrep)
    @test_throws ArgumentError e_hopping(Float64, SU2Irrep, SU2Irrep)
    I2 = sectortype(t)
    even = I2(0, 1 // 2, 0)
    odd = I2(1, 0, 1 // 2)
    f1 = only(fusiontrees((odd, odd), one(I2)))
    f2 = only(fusiontrees((even, even), one(I2)))
    f3 = only(fusiontrees((even, odd), I2((1, 1 // 2, 1 // 2))))
    f4 = only(fusiontrees((odd, even), I2((1, 1 // 2, 1 // 2))))
    @test all(t[f1, f2] .≈ 2im) && all(t[f2, f1] .≈ -2im)
    @test all(t[f3, f4] .≈ im) && all(t[f4, f3] .≈ -im)

    # e_plus_e_min moves a particle from the second site to the first:
    # ⟨↑,0|e⁺e⁻|0,↑⟩ = 1 while ⟨0,↑|e⁺e⁻|↑,0⟩ = 0, in every symmetric version
    # (regression check: the hand-written SU2-spin versions used to be transposed)
    for (P, S) in Iterators.product(particle_syms, spin_syms)
        has_e_pm(P, S) || continue
        A = convert(Array, e_plus_e_min(ComplexF64, P, S))
        U = convert(Array, basis_transform(P, S))
        i0 = findfirst(==(1), U[:, 1]) # dense index of |0⟩
        iu = findfirst(==(1), U[:, 3]) # dense index of |↑⟩
        @test A[iu, i0, i0, iu] ≈ 1
        @test abs(A[i0, iu, iu, i0]) < 1.0e-12
    end
end

@testset "basic properties" begin
    for (particle_symmetry, spin_symmetry) in Iterators.product(particle_syms, spin_syms)
        # test hopping operator
        if has_e_pm(particle_symmetry, spin_symmetry)
            epem = e_plus_e_min(particle_symmetry, spin_symmetry)
            emep = e_min_e_plus(particle_symmetry, spin_symmetry)
            @test epem' ≈ -emep ≈ swap_2sites(epem)
        else
            @test_throws ArgumentError e_plus_e_min(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError e_min_e_plus(particle_symmetry, spin_symmetry)
        end
        ehop = e_hopping(particle_symmetry, spin_symmetry)
        @test ehop' ≈ ehop

        if has_u_num(particle_symmetry, spin_symmetry)
            dpdm = d_plus_d_min(particle_symmetry, spin_symmetry)
            dmdp = d_min_d_plus(particle_symmetry, spin_symmetry)
            @test dpdm' ≈ -dmdp ≈ swap_2sites(dpdm)
            upum = u_plus_u_min(particle_symmetry, spin_symmetry)
            umup = u_min_u_plus(particle_symmetry, spin_symmetry)
            @test upum' ≈ -umup ≈ swap_2sites(upum)
        else
            @test_throws ArgumentError u_plus_u_min(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError u_min_u_plus(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError d_plus_d_min(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError d_min_d_plus(particle_symmetry, spin_symmetry)
        end

        # test number operator
        if has_u_num(particle_symmetry, spin_symmetry)
            @test e_num(particle_symmetry, spin_symmetry) ≈
                u_num(particle_symmetry, spin_symmetry) +
                d_num(particle_symmetry, spin_symmetry)
            @test ud_num(particle_symmetry, spin_symmetry) ≈
                u_num(particle_symmetry, spin_symmetry) *
                d_num(particle_symmetry, spin_symmetry) ≈
                d_num(particle_symmetry, spin_symmetry) *
                u_num(particle_symmetry, spin_symmetry)
        end
        if has_e_num(particle_symmetry, spin_symmetry)
            @test half_ud_num(particle_symmetry, spin_symmetry) ≈
                ud_num(particle_symmetry, spin_symmetry) -
                e_num(particle_symmetry, spin_symmetry) / 2 +
                id(hubbard_space(particle_symmetry, spin_symmetry)) / 4
        else
            @test_throws ArgumentError e_num(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError ud_num(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError h_num(particle_symmetry, spin_symmetry)
        end

        # test singlet operators
        if has_pair(particle_symmetry, spin_symmetry)
            singm = singlet_min(particle_symmetry, spin_symmetry)
            umdm = u_min_d_min(particle_symmetry, spin_symmetry)
            dmum = d_min_u_min(particle_symmetry, spin_symmetry)
            @test swap_2sites(umdm) ≈ -dmum
            @test swap_2sites(singm) ≈ singm
            @test singm ≈ (-umdm + dmum) / sqrt(2)
            updp = u_plus_d_plus(particle_symmetry, spin_symmetry)
            dpup = d_plus_u_plus(particle_symmetry, spin_symmetry)
            @test swap_2sites(updp) ≈ -dpup
        elseif has_singlet(particle_symmetry, spin_symmetry)
            singm = singlet_min(particle_symmetry, spin_symmetry)
            @test swap_2sites(singm) ≈ singm
        else
            @test_throws ArgumentError singlet_plus(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError singlet_min(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError u_min_d_min(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError d_min_u_min(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError u_plus_d_plus(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError d_plus_u_plus(particle_symmetry, spin_symmetry)
        end

        # test triplet operators
        if has_triplet(particle_symmetry, spin_symmetry)
            umum = u_min_u_min(particle_symmetry, spin_symmetry)
            dmdm = d_min_d_min(particle_symmetry, spin_symmetry)
            upup = u_plus_u_plus(particle_symmetry, spin_symmetry)
            dpdp = d_plus_d_plus(particle_symmetry, spin_symmetry)
            @test swap_2sites(umum) ≈ -umum
            @test swap_2sites(dmdm) ≈ -dmdm
            @test swap_2sites(upup) ≈ -upup
            @test swap_2sites(dpdp) ≈ -dpdp
        else
            @test_throws ArgumentError u_min_u_min(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError d_min_d_min(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError u_plus_u_plus(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError d_plus_d_plus(particle_symmetry, spin_symmetry)
        end

        # test spin operator
        if has_spin_ops(particle_symmetry, spin_symmetry)
            test_spin_algebra(
                S_x(particle_symmetry, spin_symmetry),
                S_y(particle_symmetry, spin_symmetry),
                S_z(particle_symmetry, spin_symmetry),
            )
            # test S_plus and S_min
            @test S_plus_S_min(particle_symmetry, spin_symmetry) ≈
                S_plus(particle_symmetry, spin_symmetry) ⊗
                S_min(particle_symmetry, spin_symmetry)
            @test S_min_S_plus(particle_symmetry, spin_symmetry) ≈
                S_min(particle_symmetry, spin_symmetry) ⊗
                S_plus(particle_symmetry, spin_symmetry)
        else
            @test_throws ArgumentError S_plus(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError S_min(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError S_x(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError S_y(particle_symmetry, spin_symmetry)
            if !has_S_z(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError S_z(particle_symmetry, spin_symmetry)
            end
        end
    end
end

function hubbard_hamiltonian(particle_symmetry, spin_symmetry; t, U, mu)
    @assert length(t) + 1 == length(U) == length(mu)
    hopping = e_hopping(particle_symmetry, spin_symmetry)
    interaction = ud_num(particle_symmetry, spin_symmetry)
    chemical_potential = e_num(particle_symmetry, spin_symmetry)
    H = operator_sum(hopping, -t) +
        operator_sum(interaction, U) +
        operator_sum(chemical_potential, -mu)

    return H
end
# particle-hole symmetric Hamiltonian (mu = U/2), compatible with all symmetries
function hubbard_hamiltonian_ph(particle_symmetry, spin_symmetry; t, U)
    @assert length(t) + 1 == length(U)
    hopping = e_hopping(particle_symmetry, spin_symmetry)
    interaction = half_ud_num(particle_symmetry, spin_symmetry)
    H = operator_sum(hopping, -t) +
        operator_sum(interaction, U)
    return H
end

@testset "spectrum" begin
    rng = StableRNG(123)
    L = 4
    t = randn(rng, L - 1)
    U = randn(rng, L)

    # particle-hole symmetric spectrum across all symmetry combinations: a many-body
    # integration check that the staggered η-gauge e_hopping (verified element-wise per-bond
    # above) assembles correctly into a chain under SU2 particle symmetry.
    H_triv = hubbard_hamiltonian_ph(Trivial, Trivial; t, U)
    vals_triv = expanded_eigenvalues(H_triv)

    for (particle_symmetry, spin_symmetry) in Iterators.product(particle_syms, spin_syms)
        particle_symmetry == spin_symmetry == Trivial && continue
        H_symm = hubbard_hamiltonian_ph(particle_symmetry, spin_symmetry; t, U)
        vals_symm = expanded_eigenvalues(H_symm)
        @test vals_triv ≈ vals_symm
    end
end

@testset "Exact diagonalisation" begin
    for (particle_symmetry, spin_symmetry) in Iterators.product(particle_syms, spin_syms)
        has_e_num(particle_symmetry, spin_symmetry) || continue
        rng = StableRNG(123)

        L = 2
        t, U = rand(rng, 5)
        mu = 0.0
        E⁻ = U / 2 - sqrt((U / 2)^2 + 4 * t^2)
        E⁺ = U / 2 + sqrt((U / 2)^2 + 4 * t^2)
        H_triv = hubbard_hamiltonian(
            particle_symmetry, spin_symmetry;
            t = fill(t, L - 1), U = fill(U, L), mu = fill(mu, L)
        )

        # Values based on https://arxiv.org/pdf/0807.4878. Introduction to Hubbard Model and Exact Diagonalization
        true_eigenvals = sort(
            vcat(fill(-t, 2), E⁻, fill(0, 4), fill(t, 2), fill(U - t, 2), U, E⁺, fill(U + t, 2), 2 * U)
        )
        eigenvals = expanded_eigenvalues(H_triv)
        @test eigenvals ≈ true_eigenvals
    end
end
