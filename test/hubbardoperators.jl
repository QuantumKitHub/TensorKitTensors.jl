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

            O = c_plus_c_min(ComplexF64, particle_symmetry, spin_symmetry)
            O_triv = c_plus_c_min(ComplexF64, Trivial, Trivial)
            test_operator(O, O_triv)

            O = c_num(ComplexF64, particle_symmetry, spin_symmetry)
            O_triv = c_num(ComplexF64, Trivial, Trivial)
            test_operator(O, O_triv)

            O = ud_num(ComplexF64, particle_symmetry, spin_symmetry)
            O_triv = ud_num(ComplexF64, Trivial, Trivial)
            test_operator(O, O_triv)
        else
            @test_broken c_plus_c_min(ComplexF64, particle_symmetry, spin_symmetry)
            @test_broken c_num(ComplexF64, particle_symmetry, spin_symmetry)
            @test_broken ud_num(ComplexF64, particle_symmetry, spin_symmetry)
        end
    end
end

@testset "basic properties" begin
    for particle_symmetry in (Trivial, U1Irrep, SU2Irrep),
        spin_symmetry in (Trivial, U1Irrep, SU2Irrep)

        if (particle_symmetry, spin_symmetry) in implemented_symmetries
            # test hermiticity
            @test c_plus_c_min(particle_symmetry, spin_symmetry)' ≈
                  c_min_c_plus(particle_symmetry, spin_symmetry)
            if spin_symmetry !== SU2Irrep
                @test d_plus_d_min(particle_symmetry, spin_symmetry)' ≈
                      d_min_d_plus(particle_symmetry, spin_symmetry)
                @test u_plus_u_min(particle_symmetry, spin_symmetry)' ≈
                      u_min_u_plus(particle_symmetry, spin_symmetry)
                @test d_plus_d_min(particle_symmetry, spin_symmetry)' ≈
                      d_min_d_plus(particle_symmetry, spin_symmetry)
                @test u_plus_u_min(particle_symmetry, spin_symmetry)' ≈
                      u_min_u_plus(particle_symmetry, spin_symmetry)
            end

            # test number operator
            if spin_symmetry !== SU2Irrep
                @test c_num(particle_symmetry, spin_symmetry) ≈
                      u_num(particle_symmetry, spin_symmetry) +
                      d_num(particle_symmetry, spin_symmetry)
                @test ud_num(particle_symmetry, spin_symmetry) ≈
                      u_num(particle_symmetry, spin_symmetry) *
                      d_num(particle_symmetry, spin_symmetry) ≈
                      d_num(particle_symmetry, spin_symmetry) *
                      u_num(particle_symmetry, spin_symmetry)
            else
                @test_broken u_plus_u_min(particle_symmetry, spin_symmetry)
                @test_broken d_plus_d_min(particle_symmetry, spin_symmetry)
            end
        else
            @test_broken c_plus_c_min(particle_symmetry, spin_symmetry)
            @test_broken c_min_c_plus(particle_symmetry, spin_symmetry)
            @test_broken d_plus_d_min(particle_symmetry, spin_symmetry)
            @test_broken u_plus_u_min(particle_symmetry, spin_symmetry)
        end
    end
end

function hubbard_hamiltonian(particle_symmetry, spin_symmetry; t, U, mu, L)
    hopping = t * (c_plus_c_min(particle_symmetry, spin_symmetry) +
                   c_min_c_plus(particle_symmetry, spin_symmetry))
    interaction = U * ud_num(particle_symmetry, spin_symmetry)
    chemical_potential = mu * c_num(particle_symmetry, spin_symmetry)
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
