using TensorKit
using LinearAlgebra: eigvals
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.TJOperators
using StableRNGs

particle_syms = (Trivial, U1Irrep)
spin_syms = (Trivial, U1Irrep, SU2Irrep)

@testset "type inference" begin
    @test (@testinferred S_exchange()) isa AbstractTensorMap
    @test (@testinferred S_exchange(U1Irrep, SU2Irrep)) isa AbstractTensorMap
    @test (@testinferred S_exchange(Float64, U1Irrep, SU2Irrep)) isa AbstractTensorMap
    @test (@testinferred S_exchange(U1Irrep, SU2Irrep; slave_fermion = true)) isa AbstractTensorMap
    @test (@testinferred S_exchange(Float64, U1Irrep, SU2Irrep; slave_fermion = true)) isa AbstractTensorMap
    @test (@testinferred e_hopping(U1Irrep, U1Irrep; slave_fermion = true)) isa AbstractTensorMap
    @test (@testinferred e_hopping(Float64, U1Irrep, U1Irrep; slave_fermion = true)) isa AbstractTensorMap
end

@testset "Compare symmetric with trivial tensors" begin
    L = 4
    for (particle_symmetry, spin_symmetry, slave_fermion) in Iterators.product(particle_syms, spin_syms, (false, true))
        space = tj_space(particle_symmetry, spin_symmetry; slave_fermion)

        O = e_plus_e_min(
            ComplexF64, particle_symmetry, spin_symmetry;
            slave_fermion
        )
        O_triv = e_plus_e_min(ComplexF64, Trivial, Trivial; slave_fermion)
        test_operator(O, O_triv; L)

        O = e_num(ComplexF64, particle_symmetry, spin_symmetry; slave_fermion)
        O_triv = e_num(ComplexF64, Trivial, Trivial; slave_fermion)
        test_operator(O, O_triv; L)

        O = S_exchange(ComplexF64, particle_symmetry, spin_symmetry; slave_fermion)
        O_triv = S_exchange(ComplexF64, Trivial, Trivial; slave_fermion)
        test_operator(O, O_triv; L)

        if particle_symmetry == Trivial
            O = singlet_plus(ComplexF64, particle_symmetry, spin_symmetry; slave_fermion)
            O_triv = singlet_plus(ComplexF64, Trivial, Trivial; slave_fermion)
            test_operator(O, O_triv; L)
        end
    end
end

@testset "basic properties" begin
    for (particle_symmetry, spin_symmetry, slave_fermion) in Iterators.product(particle_syms, spin_syms, (false, true))
        # test hopping operator
        epem = e_plus_e_min(particle_symmetry, spin_symmetry; slave_fermion)
        emep = e_min_e_plus(particle_symmetry, spin_symmetry; slave_fermion)
        @test epem' ≈ -emep ≈ swap_2sites(epem)
        if spin_symmetry !== SU2Irrep
            dpdm = d_plus_d_min(particle_symmetry, spin_symmetry)
            dmdp = d_min_d_plus(particle_symmetry, spin_symmetry)
            @test dpdm' ≈ -dmdp ≈ swap_2sites(dpdm)
            upum = u_plus_u_min(particle_symmetry, spin_symmetry)
            umup = u_min_u_plus(particle_symmetry, spin_symmetry)
            @test upum' ≈ -umup ≈ swap_2sites(upum)
        else
            @test_throws ArgumentError d_plus_d_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError d_min_d_plus(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError u_plus_u_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError u_min_u_plus(particle_symmetry, spin_symmetry; slave_fermion)
        end

        # test number operator
        if spin_symmetry !== SU2Irrep
            pspace = tj_space(particle_symmetry, spin_symmetry; slave_fermion)
            @test e_num(particle_symmetry, spin_symmetry; slave_fermion) ≈
                u_num(particle_symmetry, spin_symmetry; slave_fermion) +
                d_num(particle_symmetry, spin_symmetry; slave_fermion)
            @test u_num(particle_symmetry, spin_symmetry; slave_fermion) *
                d_num(particle_symmetry, spin_symmetry; slave_fermion) ≈
                d_num(particle_symmetry, spin_symmetry; slave_fermion) *
                u_num(particle_symmetry, spin_symmetry; slave_fermion) ≈
                zeros(pspace ← pspace)
            @test TensorKit.id(pspace) ≈
                h_num(particle_symmetry, spin_symmetry; slave_fermion) +
                e_num(particle_symmetry, spin_symmetry; slave_fermion)
        else
            @test_throws ArgumentError u_num(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError d_num(particle_symmetry, spin_symmetry; slave_fermion)
        end

        # test singlet operators
        if particle_symmetry == Trivial
            singm = singlet_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test swap_2sites(singm) ≈ singm
            if spin_symmetry !== SU2Irrep
                umdm = u_min_d_min(particle_symmetry, spin_symmetry; slave_fermion)
                dmum = d_min_u_min(particle_symmetry, spin_symmetry; slave_fermion)
                @test swap_2sites(umdm) ≈ -dmum

                @test singm ≈ (-umdm + dmum) / sqrt(2)
                updp = u_plus_d_plus(particle_symmetry, spin_symmetry; slave_fermion)
                dpup = d_plus_u_plus(particle_symmetry, spin_symmetry; slave_fermion)
                @test swap_2sites(updp) ≈ -dpup
            end
        else
            @test_throws ArgumentError singlet_plus(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError singlet_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError u_min_d_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError d_min_u_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError u_plus_d_plus(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError d_plus_u_plus(particle_symmetry, spin_symmetry; slave_fermion)
        end

        # test triplet operators
        if particle_symmetry == Trivial && spin_symmetry == Trivial
            umum = u_min_u_min(particle_symmetry, spin_symmetry; slave_fermion)
            dmdm = d_min_d_min(particle_symmetry, spin_symmetry; slave_fermion)
            upup = u_plus_u_plus(particle_symmetry, spin_symmetry; slave_fermion)
            dpdp = d_plus_d_plus(particle_symmetry, spin_symmetry; slave_fermion)
            @test swap_2sites(umum) ≈ -umum
            @test swap_2sites(dmdm) ≈ -dmdm
            @test swap_2sites(upup) ≈ -upup
            @test swap_2sites(dpdp) ≈ -dpdp
        else
            @test_throws ArgumentError u_min_u_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError d_min_d_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError u_plus_u_plus(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError d_plus_d_plus(particle_symmetry, spin_symmetry; slave_fermion)
        end

        # test spin operator
        if spin_symmetry == Trivial
            test_spin_algebra(
                S_x(particle_symmetry, spin_symmetry; slave_fermion),
                S_y(particle_symmetry, spin_symmetry; slave_fermion),
                S_z(particle_symmetry, spin_symmetry; slave_fermion),
            )
            # test S_plus and S_min
            @test S_plus_S_min(particle_symmetry, spin_symmetry; slave_fermion) ≈
                S_plus(particle_symmetry, spin_symmetry; slave_fermion) ⊗
                S_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test S_min_S_plus(particle_symmetry, spin_symmetry; slave_fermion) ≈
                S_min(particle_symmetry, spin_symmetry; slave_fermion) ⊗
                S_plus(particle_symmetry, spin_symmetry; slave_fermion)
        else
            @test_throws ArgumentError S_plus(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError S_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError S_x(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError S_y(particle_symmetry, spin_symmetry; slave_fermion)
            if spin_symmetry != U1Irrep
                @test_throws ArgumentError S_z(particle_symmetry, spin_symmetry; slave_fermion)
            end
        end
    end
end

function tjhamiltonian(particle_symmetry, spin_symmetry; t, J, mu, L, slave_fermion)
    num = e_num(particle_symmetry, spin_symmetry; slave_fermion)
    hop_heis = (-t) * e_hopping(particle_symmetry, spin_symmetry; slave_fermion) +
        J * (S_exchange(particle_symmetry, spin_symmetry; slave_fermion) - (1 / 4) * (num ⊗ num))
    chemical_potential = (-mu) * num
    H = operator_sum(hop_heis, L) + operator_sum(chemical_potential, L)
    return H
end

@testset "spectrum" begin
    rng = StableRNG(123)
    L = 4
    for slave_fermion in (false, true)
        t, J, mu = rand(rng, 3)
        H_triv = tjhamiltonian(Trivial, Trivial; t, J, mu, L, slave_fermion)
        vals_triv = expanded_eigenvalues(H_triv)

        for (particle_symmetry, spin_symmetry) in Iterators.product(particle_syms, spin_syms)
            (particle_symmetry, spin_symmetry) == (Trivial, Trivial) && continue
            H_symm = tjhamiltonian(
                particle_symmetry, spin_symmetry; t, J, mu, L, slave_fermion
            )
            vals_symm = expanded_eigenvalues(H_symm)
            @test vals_triv ≈ vals_symm
        end
    end
end

@testset "Exact Diagonalisation" begin
    rng = StableRNG(123)
    t, J = rand(rng, 2)
    true_eigenvals = sort(vcat(-J, fill(-t, 2), fill(t, 2), fill(0.0, 4)))
    for (P, S, slave_fermion) in Iterators.product(particle_syms, spin_syms, (false, true))
        H = tjhamiltonian(P, S; t, J, mu = 0.0, L = 2, slave_fermion)
        eigenvals = expanded_eigenvalues(H)
        @test eigenvals ≈ true_eigenvals
    end
end
