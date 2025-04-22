using TensorKit
using LinearAlgebra: tr, eigvals
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.TJOperators
using StableRNGs

implemented_symmetries = [(Trivial, Trivial), (Trivial, U1Irrep), (Trivial, SU2Irrep),
                          (U1Irrep, Trivial), (U1Irrep, U1Irrep), (U1Irrep, SU2Irrep)]

@testset "Compare symmetric with trivial tensors" begin
    for particle_symmetry in [Trivial, U1Irrep],
        spin_symmetry in [Trivial, U1Irrep, SU2Irrep]

        for slave_fermion in (false, true)
            if (particle_symmetry, spin_symmetry) in implemented_symmetries
                space = tj_space(particle_symmetry, spin_symmetry; slave_fermion)

                O = e_plus_e_min(ComplexF64, particle_symmetry, spin_symmetry;
                                 slave_fermion)
                O_triv = e_plus_e_min(ComplexF64, Trivial, Trivial; slave_fermion)
                test_operator(O, O_triv)

                O = e_num(ComplexF64, particle_symmetry, spin_symmetry; slave_fermion)
                O_triv = e_num(ComplexF64, Trivial, Trivial; slave_fermion)
                test_operator(O, O_triv)

            else
                @test_broken e_plus_e_min(ComplexF64, particle_symmetry, spin_symmetry)
                @test_broken e_num(ComplexF64, particle_symmetry, spin_symmetry)
            end
        end
    end
end

@testset "basic properties" begin
    for slave_fermion in (false, true)
        for particle_symmetry in [Trivial, U1Irrep],
            spin_symmetry in [Trivial, U1Irrep, SU2Irrep]

            if (particle_symmetry, spin_symmetry) in implemented_symmetries
                # test hermiticity
                @test e_plus_e_min(particle_symmetry, spin_symmetry; slave_fermion)' ≈
                      -e_min_e_plus(particle_symmetry, spin_symmetry; slave_fermion)
                if spin_symmetry !== SU2Irrep
                    @test d_plus_d_min(particle_symmetry, spin_symmetry; slave_fermion)' ≈
                          d_min_d_plus(particle_symmetry, spin_symmetry; slave_fermion)
                    @test u_plus_u_min(particle_symmetry, spin_symmetry; slave_fermion)' ≈
                          u_min_u_plus(particle_symmetry, spin_symmetry; slave_fermion)
                    @test d_plus_d_min(particle_symmetry, spin_symmetry; slave_fermion)' ≈
                          d_min_d_plus(particle_symmetry, spin_symmetry; slave_fermion)
                    @test u_plus_u_min(particle_symmetry, spin_symmetry; slave_fermion)' ≈
                          u_min_u_plus(particle_symmetry, spin_symmetry; slave_fermion)
                else
                    @test_throws ArgumentError d_plus_d_min(particle_symmetry,
                                                            spin_symmetry;
                                                            slave_fermion)
                    @test_throws ArgumentError d_min_d_plus(particle_symmetry,
                                                            spin_symmetry;
                                                            slave_fermion)
                    @test_throws ArgumentError u_plus_u_min(particle_symmetry,
                                                            spin_symmetry;
                                                            slave_fermion)
                    @test_throws ArgumentError u_min_u_plus(particle_symmetry,
                                                            spin_symmetry;
                                                            slave_fermion)
                end

                # test number operator
                if spin_symmetry !== SU2Irrep
                    @test e_num(particle_symmetry, spin_symmetry; slave_fermion) ≈
                          u_num(particle_symmetry, spin_symmetry; slave_fermion) +
                          d_num(particle_symmetry, spin_symmetry; slave_fermion)
                    @test u_num(particle_symmetry, spin_symmetry; slave_fermion) *
                          d_num(particle_symmetry, spin_symmetry; slave_fermion) ≈
                          d_num(particle_symmetry, spin_symmetry; slave_fermion) *
                          u_num(particle_symmetry, spin_symmetry; slave_fermion)
                    @test TensorKit.id(tj_space(particle_symmetry, spin_symmetry;
                                                slave_fermion)) ≈
                          h_num(particle_symmetry, spin_symmetry; slave_fermion) +
                          e_num(particle_symmetry, spin_symmetry; slave_fermion)
                else
                    @test_throws ArgumentError u_num(particle_symmetry, spin_symmetry;
                                                     slave_fermion)
                    @test_throws ArgumentError d_num(particle_symmetry, spin_symmetry;
                                                     slave_fermion)
                end

                # test spin operator
                if particle_symmetry == Trivial && spin_symmetry !== SU2Irrep
                    @test singlet_min(particle_symmetry, spin_symmetry; slave_fermion) ≈
                          (u_min_d_min(particle_symmetry, spin_symmetry; slave_fermion) -
                           d_min_u_min(particle_symmetry, spin_symmetry; slave_fermion)) /
                          sqrt(2)
                else
                    @test_throws ArgumentError singlet_min(particle_symmetry, spin_symmetry;
                                                         slave_fermion)
                    @test_throws ArgumentError u_min_d_min(particle_symmetry, spin_symmetry;
                                                           slave_fermion)
                    @test_throws ArgumentError d_min_u_min(particle_symmetry, spin_symmetry;
                                                           slave_fermion)
                end

                # test hopping operator
                @test e_hopping(particle_symmetry, spin_symmetry; slave_fermion) ≈
                      e_plus_e_min(particle_symmetry, spin_symmetry; slave_fermion) -
                      e_min_e_plus(particle_symmetry, spin_symmetry; slave_fermion)

                if spin_symmetry == Trivial
                    ε = zeros(ComplexF64, 3, 3, 3)
                    for i in 1:3
                        ε[mod1(i, 3), mod1(i + 1, 3), mod1(i + 2, 3)] = 1
                        ε[mod1(i, 3), mod1(i - 1, 3), mod1(i - 2, 3)] = -1
                    end
                    Svec = [S_x(particle_symmetry, spin_symmetry; slave_fermion),
                            S_y(particle_symmetry, spin_symmetry; slave_fermion),
                            S_z(particle_symmetry, spin_symmetry; slave_fermion)]
                    # Hermiticity
                    for s in Svec
                        @test s' ≈ s
                    end
                    # operators should be normalized
                    S = 1 / 2
                    @test sum(tr(Svec[i]^2) for i in 1:3) / (2S + 1) ≈ S * (S + 1)
                    # test S_plus and S_min
                    @test S_plus_S_min(particle_symmetry, spin_symmetry; slave_fermion) ≈
                          S_plus(particle_symmetry, spin_symmetry; slave_fermion) ⊗
                          S_min(particle_symmetry, spin_symmetry; slave_fermion)
                    @test S_min_S_plus(particle_symmetry, spin_symmetry; slave_fermion) ≈
                          S_min(particle_symmetry, spin_symmetry; slave_fermion) ⊗
                          S_plus(particle_symmetry, spin_symmetry; slave_fermion)
                    # commutation relations
                    for i in 1:3, j in 1:3
                        @test Svec[i] * Svec[j] - Svec[j] * Svec[i] ≈
                              sum(im * ε[i, j, k] * Svec[k] for k in 1:3)
                    end
                end
            else
                @test_broken d_plus_d_min(particle_symmetry, spin_symmetry; slave_fermion)
                @test_broken d_min_d_plus(particle_symmetry, spin_symmetry; slave_fermion)
                @test_broken u_plus_u_min(particle_symmetry, spin_symmetry; slave_fermion)
                @test_broken u_min_u_plus(particle_symmetry, spin_symmetry; slave_fermion)
                @test_broken e_num(particle_symmetry, spin_symmetry; slave_fermion)
                @test_broken u_num(particle_symmetry, spin_symmetry; slave_fermion)
                @test_broken d_num(particle_symmetry, spin_symmetry; slave_fermion)
            end
        end
    end
end

function tjhamiltonian(particle_symmetry, spin_symmetry; t, J, mu, L, slave_fermion)
    num = e_num(particle_symmetry, spin_symmetry; slave_fermion)
    hop_heis = (-t) * (e_plus_e_min(particle_symmetry, spin_symmetry; slave_fermion) -
                       e_min_e_plus(particle_symmetry, spin_symmetry; slave_fermion)) +
               J *
               (S_exchange(particle_symmetry, spin_symmetry; slave_fermion) -
                (1 / 4) * (num ⊗ num))
    chemical_potential = (-mu) * num
    I = id(tj_space(particle_symmetry, spin_symmetry; slave_fermion))
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
                H_symm = tjhamiltonian(particle_symmetry, spin_symmetry; t, J, mu, L,
                                       slave_fermion)
                vals_symm = mapreduce(vcat, eigvals(H_symm)) do (c, v)
                    return repeat(real.(v), dim(c))
                end
                sort!(vals_symm)
                @test vals_triv ≈ vals_symm
            else
                @test_broken tjhamiltonian(particle_symmetry, spin_symmetry; t, J, mu, L,
                                           slave_fermion)
            end
        end
    end
end

@testset "Exact Diagonalisation" begin
    rng = StableRNG(123)
    L = 2

    for particle_symmetry in [Trivial, U1Irrep],
        spin_symmetry in [Trivial, U1Irrep, SU2Irrep]

        for slave_fermion in (false, true)
            if (particle_symmetry, spin_symmetry) in implemented_symmetries
                t, J = rand(rng, 2)
                num = e_num(particle_symmetry, spin_symmetry; slave_fermion)
                H = (-t) *
                    (e_plus_e_min(particle_symmetry, spin_symmetry; slave_fermion) -
                     e_min_e_plus(particle_symmetry, spin_symmetry; slave_fermion)) +
                    J *
                    (S_exchange(particle_symmetry, spin_symmetry; slave_fermion) -
                     (1 / 4) * (num ⊗ num))

                true_eigenvals = sort(vcat([-J], repeat([-t], 2), repeat([t], 2),
                                           repeat([0.0], 4)))
                eigenvals = expanded_eigenvalues(H; L)
                @test eigenvals ≈ true_eigenvals
            end
        end
    end
end
