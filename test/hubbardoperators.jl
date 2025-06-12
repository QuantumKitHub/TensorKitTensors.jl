using TensorKit
using LinearAlgebra: tr, eigvals
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.HubbardOperators
using StableRNGs

implemented_symmetries = [(Trivial, Trivial), (Trivial, U1Irrep), (Trivial, SU2Irrep),
                          (U1Irrep, Trivial), (U1Irrep, U1Irrep), (U1Irrep, SU2Irrep)]

@testset "Compare symmetric with trivial tensors" begin
    for particle_symmetry in [Trivial, U1Irrep, SU2Irrep],
        spin_symmetry in [Trivial, U1Irrep, SU2Irrep]

        if (particle_symmetry, spin_symmetry) in implemented_symmetries
            space = hubbard_space(particle_symmetry, spin_symmetry)

            O = e_plus_e_min(ComplexF64, particle_symmetry, spin_symmetry)
            O_triv = e_plus_e_min(ComplexF64, Trivial, Trivial)
            test_operator(O, O_triv)

            O = e_num(ComplexF64, particle_symmetry, spin_symmetry)
            O_triv = e_num(ComplexF64, Trivial, Trivial)
            test_operator(O, O_triv)

            O = ud_num(ComplexF64, particle_symmetry, spin_symmetry)
            O_triv = ud_num(ComplexF64, Trivial, Trivial)
            test_operator(O, O_triv)

            O = S_exchange(ComplexF64, particle_symmetry, spin_symmetry)
            O_triv = S_exchange(ComplexF64, Trivial, Trivial)
            test_operator(O, O_triv)
        else
            @test_broken e_plus_e_min(ComplexF64, particle_symmetry, spin_symmetry)
            @test_broken e_num(ComplexF64, particle_symmetry, spin_symmetry)
            @test_broken ud_num(ComplexF64, particle_symmetry, spin_symmetry)
            @test_broken S_exchange(ComplexF64, particle_symmetry, spin_symmetry)
        end
    end
end

@testset "basic properties" begin
    for particle_symmetry in (Trivial, U1Irrep, SU2Irrep),
        spin_symmetry in (Trivial, U1Irrep, SU2Irrep)

        if (particle_symmetry, spin_symmetry) in implemented_symmetries
            # test hopping operator
            epem = e_plus_e_min(particle_symmetry, spin_symmetry)
            emep = e_min_e_plus(particle_symmetry, spin_symmetry)
            @test epem' ≈ -emep ≈ swap_2sites(epem)
            if spin_symmetry !== SU2Irrep
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
            if spin_symmetry !== SU2Irrep
                @test e_num(particle_symmetry, spin_symmetry) ≈
                      u_num(particle_symmetry, spin_symmetry) +
                      d_num(particle_symmetry, spin_symmetry)
                @test ud_num(particle_symmetry, spin_symmetry) ≈
                      u_num(particle_symmetry, spin_symmetry) *
                      d_num(particle_symmetry, spin_symmetry) ≈
                      d_num(particle_symmetry, spin_symmetry) *
                      u_num(particle_symmetry, spin_symmetry)
            end

            # test singlet operators
            if particle_symmetry == Trivial && spin_symmetry !== SU2Irrep
                singm = singlet_min(particle_symmetry, spin_symmetry)
                umdm = u_min_d_min(particle_symmetry, spin_symmetry)
                dmum = d_min_u_min(particle_symmetry, spin_symmetry)
                @test swap_2sites(umdm) ≈ -dmum
                @test swap_2sites(singm) ≈ singm
                @test singm ≈ (-umdm + dmum) / sqrt(2)
                updp = u_plus_d_plus(particle_symmetry, spin_symmetry)
                dpup = d_plus_u_plus(particle_symmetry, spin_symmetry)
                @test swap_2sites(updp) ≈ -dpup
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
            if spin_symmetry == Trivial
                ε = zeros(ComplexF64, 3, 3, 3)
                for i in 1:3
                    ε[mod1(i, 3), mod1(i + 1, 3), mod1(i + 2, 3)] = 1
                    ε[mod1(i, 3), mod1(i - 1, 3), mod1(i - 2, 3)] = -1
                end
                Svec = [S_x(particle_symmetry, spin_symmetry),
                        S_y(particle_symmetry, spin_symmetry),
                        S_z(particle_symmetry, spin_symmetry)]
                # Hermiticity
                for s in Svec
                    @test s' ≈ s
                end
                # operators should be normalized
                S = 1 / 2
                @test sum(tr(Svec[i]^2) for i in 1:3) / (2S + 1) ≈ S * (S + 1)
                # test S_plus and S_min
                @test S_plus_S_min(particle_symmetry, spin_symmetry) ≈
                      S_plus(particle_symmetry, spin_symmetry) ⊗
                      S_min(particle_symmetry, spin_symmetry)
                @test S_min_S_plus(particle_symmetry, spin_symmetry) ≈
                      S_min(particle_symmetry, spin_symmetry) ⊗
                      S_plus(particle_symmetry, spin_symmetry)
                # commutation relations
                for i in 1:3, j in 1:3
                    @test Svec[i] * Svec[j] - Svec[j] * Svec[i] ≈
                          sum(im * ε[i, j, k] * Svec[k] for k in 1:3)
                end
            else
                @test_throws ArgumentError S_plus(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError S_min(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError S_x(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError S_y(particle_symmetry, spin_symmetry)
                if spin_symmetry != U1Irrep
                    @test_throws ArgumentError S_z(particle_symmetry, spin_symmetry)
                end
            end
        else
            @test_broken e_plus_e_min(particle_symmetry, spin_symmetry)
            @test_broken e_min_e_plus(particle_symmetry, spin_symmetry)
            @test_broken d_plus_d_min(particle_symmetry, spin_symmetry)
            @test_broken d_min_d_plus(particle_symmetry, spin_symmetry)
            @test_broken u_plus_u_min(particle_symmetry, spin_symmetry)
            @test_broken u_min_u_plus(particle_symmetry, spin_symmetry)
        end
    end
end

function hubbard_hamiltonian(particle_symmetry, spin_symmetry; t, U, mu, L)
    hopping = -t * e_hopping(particle_symmetry, spin_symmetry)
    interaction = U * ud_num(particle_symmetry, spin_symmetry)
    chemical_potential = -mu * e_num(particle_symmetry, spin_symmetry)
    I = id(hubbard_space(particle_symmetry, spin_symmetry))
    H = sum(1:(L - 1)) do i
            return reduce(⊗, insert!(collect(Any, fill(I, L - 2)), i, hopping))
        end +
        sum(1:L) do i
            return reduce(⊗, insert!(collect(Any, fill(I, L - 1)), i, interaction))
        end +
        sum(1:L) do i
            return reduce(⊗, insert!(collect(Any, fill(I, L - 1)), i, chemical_potential))
        end
    return H
end

@testset "spectrum" begin
    L = 4
    t = randn()
    U = randn()
    mu = randn()

    H_triv = hubbard_hamiltonian(Trivial, Trivial; t, U, mu, L)
    vals_triv = mapreduce(vcat, eigvals(H_triv)) do (c, v)
        return repeat(real.(v), dim(c))
    end
    sort!(vals_triv)

    for (particle_symmetry, spin_symmetry) in implemented_symmetries
        if (particle_symmetry, spin_symmetry) == (Trivial, Trivial)
            continue
        end
        H_symm = hubbard_hamiltonian(particle_symmetry, spin_symmetry; t, U, mu, L)
        vals_symm = mapreduce(vcat, eigvals(H_symm)) do (c, v)
            return repeat(real.(v), dim(c))
        end
        sort!(vals_symm)
        @test vals_triv ≈ vals_symm
    end
end

@testset "Exact diagonalisation" begin
    for particle_symmetry in [Trivial, U1Irrep, SU2Irrep],
        spin_symmetry in [Trivial, U1Irrep, SU2Irrep]

        if (particle_symmetry, spin_symmetry) in implemented_symmetries
            rng = StableRNG(123)

            L = 2
            t, U = rand(rng, 5)
            mu = 0.0
            E⁻ = U / 2 - sqrt((U / 2)^2 + 4 * t^2)
            E⁺ = U / 2 + sqrt((U / 2)^2 + 4 * t^2)
            H_triv = hubbard_hamiltonian(particle_symmetry, spin_symmetry; t, U, mu, L)

            # Values based on https://arxiv.org/pdf/0807.4878. Introduction to Hubbard Model and Exact Diagonalization
            true_eigenvals = sort(vcat(repeat([-t], 2), [E⁻], repeat([0], 4),
                                       repeat([t], 2),
                                       repeat([U - t], 2), [U], [E⁺], repeat([U + t], 2),
                                       [2 * U]))
            eigenvals = expanded_eigenvalues(H_triv; L)
            @test eigenvals ≈ true_eigenvals
        end
    end
end
