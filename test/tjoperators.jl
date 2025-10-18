using TensorKit
using LinearAlgebra: tr, eigvals
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.TJOperators
using StableRNGs

implemented_symmetries = [
    (Trivial, Trivial), (Trivial, U1Irrep), (Trivial, SU2Irrep),
    (U1Irrep, Trivial), (U1Irrep, U1Irrep), (U1Irrep, SU2Irrep),
]

@testset "Compare symmetric with trivial tensors" begin
    for particle_symmetry in [Trivial, U1Irrep],
            spin_symmetry in [Trivial, U1Irrep, SU2Irrep]

        if (particle_symmetry, spin_symmetry) in implemented_symmetries
            space = tj_space(particle_symmetry, spin_symmetry)

            O = e_plus_e_min(ComplexF64, particle_symmetry, spin_symmetry)
            O_triv = e_plus_e_min()
            test_operator(O, O_triv)

            O_sf = transform_slave_fermion(O)
            @test norm(O_sf) ≈ norm(O)
            @test transform_slave_fermion(O_sf) ≈ O

            O_sf_triv = transform_slave_fermion(O_triv)
            test_operator(O_sf, O_sf_triv)

            O = e_num(ComplexF64, particle_symmetry, spin_symmetry)
            O_triv = e_num()
            test_operator(O, O_triv)

            O_sf = transform_slave_fermion(O)
            @test norm(O_sf) ≈ norm(O)
            @test transform_slave_fermion(O_sf) ≈ O

            O_sf_triv = transform_slave_fermion(O_triv)
            test_operator(O_sf, O_sf_triv)

            O = S_exchange(ComplexF64, particle_symmetry, spin_symmetry)
            O_triv = S_exchange()
            test_operator(O, O_triv)

            O_sf = transform_slave_fermion(O)
            @test norm(O_sf) ≈ norm(O)
            @test transform_slave_fermion(O_sf) ≈ O

            O_sf_triv = transform_slave_fermion(O_triv)
            test_operator(O_sf, O_sf_triv)
        else
            @test_broken e_plus_e_min(
                ComplexF64, particle_symmetry, spin_symmetry
            )
            @test_broken e_num(
                ComplexF64, particle_symmetry, spin_symmetry
            )
            @test_broken S_exchange(
                ComplexF64, particle_symmetry, spin_symmetry
            )
        end
    end
end

@testset "basic properties" begin
    for particle_symmetry in [Trivial, U1Irrep],
            spin_symmetry in [Trivial, U1Irrep, SU2Irrep]

        if (particle_symmetry, spin_symmetry) in implemented_symmetries
            # test hopping operator
            epem = e_plus_e_min(particle_symmetry, spin_symmetry)
            emep = e_min_e_plus(particle_symmetry, spin_symmetry)
            @test epem' ≈ -emep ≈ swap_2sites(epem)
            @test transform_slave_fermion(epem)' ≈ -transform_slave_fermion(emep) ≈
                swap_2sites(transform_slave_fermion(epem))
            if spin_symmetry !== SU2Irrep
                dpdm = d_plus_d_min(particle_symmetry, spin_symmetry)
                dmdp = d_min_d_plus(particle_symmetry, spin_symmetry)
                @test dpdm' ≈ -dmdp ≈ swap_2sites(dpdm)
                upum = u_plus_u_min(particle_symmetry, spin_symmetry)
                umup = u_min_u_plus(particle_symmetry, spin_symmetry)
                @test upum' ≈ -umup ≈ swap_2sites(upum)
            else
                @test_throws ArgumentError d_plus_d_min(
                    particle_symmetry, spin_symmetry
                )
                @test_throws ArgumentError d_min_d_plus(
                    particle_symmetry, spin_symmetry
                )
                @test_throws ArgumentError u_plus_u_min(
                    particle_symmetry, spin_symmetry
                )
                @test_throws ArgumentError u_min_u_plus(
                    particle_symmetry, spin_symmetry
                )
            end

            # test number operator
            if spin_symmetry !== SU2Irrep
                pspace = tj_space(particle_symmetry, spin_symmetry)
                @test e_num(particle_symmetry, spin_symmetry) ≈
                    u_num(particle_symmetry, spin_symmetry) +
                    d_num(particle_symmetry, spin_symmetry)
                @test u_num(particle_symmetry, spin_symmetry) *
                    d_num(particle_symmetry, spin_symmetry) ≈
                    d_num(particle_symmetry, spin_symmetry) *
                    u_num(particle_symmetry, spin_symmetry) ≈
                    zeros(pspace ← pspace)
                @test TensorKit.id(pspace) ≈
                    h_num(particle_symmetry, spin_symmetry) +
                    e_num(particle_symmetry, spin_symmetry)
            else
                @test_throws ArgumentError u_num(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError d_num(particle_symmetry, spin_symmetry)
            end

            # test singlet operators
            if particle_symmetry == Trivial && spin_symmetry !== SU2Irrep
                singm = singlet_min(particle_symmetry, spin_symmetry)
                umdm = u_min_d_min(particle_symmetry, spin_symmetry)
                dmum = d_min_u_min(particle_symmetry, spin_symmetry)
                @test swap_2sites(umdm) ≈ -dmum
                @test swap_2sites(transform_slave_fermion(umdm)) ≈ -transform_slave_fermion(dmum)
                @test swap_2sites(singm) ≈ singm
                @test swap_2sites(transform_slave_fermion(singm)) ≈ transform_slave_fermion(singm)
                @test singm ≈ (-umdm + dmum) / sqrt(2)
                updp = u_plus_d_plus(particle_symmetry, spin_symmetry)
                dpup = d_plus_u_plus(particle_symmetry, spin_symmetry)
                @test swap_2sites(updp) ≈ -dpup
                @test swap_2sites(transform_slave_fermion(updp)) ≈ -transform_slave_fermion(dpup)

            else
                @test_throws ArgumentError singlet_plus(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError singlet_min(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError u_min_d_min(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError d_min_u_min(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError u_plus_d_plus(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError d_plus_u_plus(particle_symmetry, spin_symmetry)
            end

            # test triplet operators
            if particle_symmetry == Trivial && spin_symmetry == Trivial
                umum = u_min_u_min(particle_symmetry, spin_symmetry)
                dmdm = d_min_d_min(particle_symmetry, spin_symmetry)
                upup = u_plus_u_plus(particle_symmetry, spin_symmetry)
                dpdp = d_plus_d_plus(particle_symmetry, spin_symmetry)
                for O in (umum, dmdm, upup, dpdp)
                    @test swap_2sites(O) ≈ -O
                    @test swap_2sites(transform_slave_fermion(O)) ≈ -transform_slave_fermion(O)
                end
            else
                @test_throws ArgumentError u_min_u_min(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError d_min_d_min(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError u_plus_u_plus(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError d_plus_d_plus(particle_symmetry, spin_symmetry)
            end

            # test spin operator
            if spin_symmetry == Trivial
                ε = zeros(ComplexF64, 3, 3, 3)
                for i in 1:3
                    ε[mod1(i, 3), mod1(i + 1, 3), mod1(i + 2, 3)] = 1
                    ε[mod1(i, 3), mod1(i - 1, 3), mod1(i - 2, 3)] = -1
                end
                Svec = [
                    S_x(particle_symmetry, spin_symmetry),
                    S_y(particle_symmetry, spin_symmetry),
                    S_z(particle_symmetry, spin_symmetry),
                ]
                Svec_sf = map(transform_slave_fermion, Svec)

                # Hermiticity
                for s in Svec
                    @test s' ≈ s
                end
                for s in Svec_sf
                    @test s' ≈ s
                end

                # operators should be normalized
                S = 1 / 2
                @test sum(tr(Svec[i]^2) for i in 1:3) / (2S + 1) ≈ S * (S + 1)
                @test sum(tr(Svec_sf[i]^2) for i in 1:3) / (2S + 1) ≈ S * (S + 1)

                # test S_plus and S_min
                Sp = S_plus(particle_symmetry, spin_symmetry)
                Sm = S_min(particle_symmetry, spin_symmetry)
                Spm = S_plus_S_min(particle_symmetry, spin_symmetry)
                Smp = S_min_S_plus(particle_symmetry, spin_symmetry)
                @test Spm ≈ Sp ⊗ Sm
                @test transform_slave_fermion(Spm) ≈ transform_slave_fermion(Sp) ⊗ transform_slave_fermion(Sm)
                @test Smp ≈ Sm ⊗ Sp
                @test transform_slave_fermion(Smp) ≈ transform_slave_fermion(Sm) ⊗ transform_slave_fermion(Sp)

                # commutation relations
                for i in 1:3, j in 1:3
                    @test Svec[i] * Svec[j] - Svec[j] * Svec[i] ≈
                        sum(im * ε[i, j, k] * Svec[k] for k in 1:3)
                end
                for i in 1:3, j in 1:3
                    @test Svec_sf[i] * Svec_sf[j] - Svec_sf[j] * Svec_sf[i] ≈
                        sum(im * ε[i, j, k] * Svec_sf[k] for k in 1:3)
                end
            else
                @test_throws ArgumentError S_plus(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError S_min(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError S_x(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError S_y(particle_symmetry, spin_symmetry)
                spin_symmetry == U1Irrep ||
                    @test_throws ArgumentError S_z(particle_symmetry, spin_symmetry)
            end
        else
            @test_broken d_plus_d_min(particle_symmetry, spin_symmetry)
            @test_broken d_min_d_plus(particle_symmetry, spin_symmetry)
            @test_broken u_plus_u_min(particle_symmetry, spin_symmetry)
            @test_broken u_min_u_plus(particle_symmetry, spin_symmetry)
            @test_broken e_num(particle_symmetry, spin_symmetry)
            @test_broken u_num(particle_symmetry, spin_symmetry)
            @test_broken d_num(particle_symmetry, spin_symmetry)
        end
    end
end

function tjhamiltonian(particle_symmetry, spin_symmetry; t, J, mu, L, slave_fermion)
    num = e_num(particle_symmetry, spin_symmetry)
    hop_heis = (-t) * e_hopping(particle_symmetry, spin_symmetry) + J * (S_exchange(particle_symmetry, spin_symmetry) - (1 / 4) * (num ⊗ num))
    chemical_potential = (-mu) * num
    I = id(tj_space(particle_symmetry, spin_symmetry))

    if slave_fermion
        hop_heis, chemical_potential, I = transform_slave_fermion.((hop_heis, chemical_potential, I))
    end
    H = sum(1:(L - 1)) do i
        return reduce(⊗, insert!(collect(Any, fill(I, L - 2)), i, hop_heis))
    end + sum(1:L) do i
        return reduce(⊗, insert!(collect(Any, fill(I, L - 1)), i, chemical_potential))
    end
    return H
end

@testset "spectrum" begin
    rng = StableRNG(123)
    L = 4

    for slave_fermion in (false, true)
        t, J, mu = rand(rng, 3)
        H_triv = tjhamiltonian(Trivial, Trivial; t, J, mu, L, slave_fermion)
        vals_triv = mapreduce(vcat, eigvals(H_triv)) do (c, v)
            return repeat(real.(v), dim(c))
        end
        sort!(vals_triv)

        for particle_symmetry in (Trivial, U1Irrep),
                spin_symmetry in (Trivial, U1Irrep, SU2Irrep)

            if (particle_symmetry, spin_symmetry) in implemented_symmetries
                if (particle_symmetry, spin_symmetry) == (Trivial, Trivial)
                    continue
                end
                H_symm = tjhamiltonian(
                    particle_symmetry, spin_symmetry; t, J, mu, L, slave_fermion
                )
                vals_symm = mapreduce(vcat, eigvals(H_symm)) do (c, v)
                    return repeat(real.(v), dim(c))
                end
                sort!(vals_symm)
                @test vals_triv ≈ vals_symm
            else
                @test_broken tjhamiltonian(
                    particle_symmetry, spin_symmetry; t, J, mu, L, slave_fermion
                )
            end
        end
    end
end

@testset "Exact Diagonalisation" begin
    rng = StableRNG(123)
    L = 2

    for particle_symmetry in [Trivial, U1Irrep],
            spin_symmetry in [Trivial, U1Irrep, SU2Irrep]

        if (particle_symmetry, spin_symmetry) in implemented_symmetries
            t, J = rand(rng, 2)
            num = e_num(particle_symmetry, spin_symmetry)
            H = (-t) * e_hopping(particle_symmetry, spin_symmetry) +
                J * (
                S_exchange(particle_symmetry, spin_symmetry) -
                    (1 / 4) * (num ⊗ num)
            )

            true_eigenvals = sort(
                vcat([-J], repeat([-t], 2), repeat([t], 2), repeat([0.0], 4))
            )
            eigenvals = expanded_eigenvalues(H; L)
            @test eigenvals ≈ true_eigenvals
            eigenvals = expanded_eigenvalues(transform_slave_fermion(H); L)
            @test eigenvals ≈ true_eigenvals
        end
    end
end
