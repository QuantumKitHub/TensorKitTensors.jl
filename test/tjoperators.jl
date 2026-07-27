using TensorKit
using LinearAlgebra: eigvals
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors: desymmetrize
using TensorKitTensors.TJOperators
import TensorKitTensors.HubbardOperators as HO
using StableRNGs

particle_syms = (Trivial, U1Irrep)
spin_syms = (Trivial, U1Irrep, SU2Irrep)
bases = (false, true)
symmetries = Iterators.product(particle_syms, spin_syms)
symmetries_bases = Iterators.product(particle_syms, spin_syms, bases)

# operator availability, as determined by the symmetries each operator breaks:
# - u_num, d_num, S_z and the u/d hopping and spin-flip pairs break SU2 spin symmetry
# - S_x, S_y, S_plus, S_min only exist for trivial spin symmetry
# - the pair (electron-number non-conserving) operators break U1 particle symmetry, and the
#   triplet ones additionally break U1 spin symmetry
# - e_num, h_num, e_plus_e_min, e_hopping, S_exchange and the singlet-pair correlators are
#   compatible with every symmetry combination of the t-J model
has_u_num(P, S) = S !== SU2Irrep
has_S_z(P, S) = S !== SU2Irrep
has_spin_ops(P, S) = S === Trivial
has_pair(P, S) = P === Trivial && S !== SU2Irrep
has_singlet(P, S) = P === Trivial
has_triplet(P, S) = P === Trivial && S === Trivial

# operators available for all supported symmetry combinations
const ALWAYS = (e_num, h_num, e_plus_e_min, e_min_e_plus, e_hopping, S_exchange, Δ⁺ij_Δjk, Δ⁺ij_Δkl)

@testset "basis transformations" begin
    for (P, S, slave_fermion) in symmetries_bases
        U = basis_transform(P, S; slave_fermion)
        @test U isa AbstractTensorMap{Int}
        @test U' * U == one(U)
        @test U * U' == one(U)
        @test space(U) == (
            desymmetrize(tj_space(P, S; slave_fermion)) ←
                desymmetrize(tj_space(Trivial, Trivial; slave_fermion))
        )
    end
    # the reference basis of each of the two bases is its own dense order
    for slave_fermion in bases
        U = basis_transform(Trivial, Trivial; slave_fermion)
        @test U == one(U)
    end

    # the t-J model has no η-pairing doublet, and hence no SU2 particle symmetry
    @test_throws ArgumentError tj_space(SU2Irrep, Trivial)
    @test_throws ArgumentError tj_space(Trivial, Z2Irrep)
    @test_throws ArgumentError e_num(ComplexF64, SU2Irrep, SU2Irrep)
    @test_throws ArgumentError S_exchange(ComplexF64, SU2Irrep, Trivial; slave_fermion = true)

    # real and wide scalar types are preserved, in both bases
    for slave_fermion in bases
        @test scalartype(u_num(Float64, U1Irrep, U1Irrep; slave_fermion)) === Float64
        @test scalartype(S_exchange(Float64, U1Irrep, SU2Irrep; slave_fermion)) === Float64
        @test scalartype(e_hopping(Complex{BigFloat}, U1Irrep, SU2Irrep; slave_fermion)) ===
            Complex{BigFloat}
        N_big = u_num(BigFloat, U1Irrep, U1Irrep; slave_fermion)
        @test all(((c, b),) -> all(isinteger, b), blocks(N_big))
    end
end

@testset "type inference" begin
    @test (@testinferred S_exchange()) isa AbstractTensorMap
    @test (@testinferred S_exchange(Float64)) isa AbstractTensorMap
    @test (@testinferred S_exchange(U1Irrep, SU2Irrep)) isa AbstractTensorMap
    @test (@testinferred S_exchange(Float64, U1Irrep, SU2Irrep)) isa AbstractTensorMap
    @test (@testinferred S_exchange(U1Irrep, SU2Irrep; slave_fermion = true)) isa AbstractTensorMap
    @test (@testinferred S_exchange(Float64, U1Irrep, SU2Irrep; slave_fermion = true)) isa AbstractTensorMap
    @test (@testinferred e_hopping(U1Irrep, U1Irrep; slave_fermion = true)) isa AbstractTensorMap
    @test (@testinferred e_hopping(Float64, U1Irrep, U1Irrep; slave_fermion = true)) isa AbstractTensorMap
end

@testset "Compare symmetric with trivial tensors" begin
    for (particle_symmetry, spin_symmetry, slave_fermion) in symmetries_bases
        space = @testinferred tj_space(particle_symmetry, spin_symmetry; slave_fermion)
        @test dim(space) == 3

        # element-wise comparison in the dense basis catches transposes and gauge errors
        # that the spectral `test_operator` is blind to
        U = basis_transform(particle_symmetry, spin_symmetry; slave_fermion)

        for (available, fs) in (
                (true, ALWAYS),
                (has_u_num(particle_symmetry, spin_symmetry), (u_num, d_num)),
                (has_S_z(particle_symmetry, spin_symmetry), (S_z, S_plus_S_min, S_min_S_plus)),
                (has_spin_ops(particle_symmetry, spin_symmetry), (S_x, S_plus, S_min)),
                (has_pair(particle_symmetry, spin_symmetry), (u_min_d_min, d_min_u_min, u_plus_d_plus, d_plus_u_plus)),
                (has_singlet(particle_symmetry, spin_symmetry), (singlet_plus, singlet_min)),
                (has_triplet(particle_symmetry, spin_symmetry), (u_min_u_min, d_min_d_min, u_plus_u_plus, d_plus_d_plus)),
            )
            for f in fs
                if available
                    O = f(ComplexF64, particle_symmetry, spin_symmetry; slave_fermion)
                    O_triv = f(ComplexF64, Trivial, Trivial; slave_fermion)
                    test_operator_dense(O, O_triv, U)
                else
                    @test_throws ArgumentError f(
                        ComplexF64, particle_symmetry, spin_symmetry; slave_fermion
                    )
                end
            end
        end
    end
end

@testset "slave-fermion basis" begin
    # the reference operators are transformed to the slave-fermion basis before they are
    # symmetrized, which has to agree with transforming the symmetric operator itself
    for (particle_symmetry, spin_symmetry) in symmetries
        for f in (
                ALWAYS..., u_num, d_num, S_z, S_plus_S_min, S_min_S_plus, S_x, S_y, S_plus,
                S_min, u_min_d_min, d_min_u_min, u_plus_d_plus, d_plus_u_plus, singlet_plus,
                singlet_min, u_min_u_min, d_min_d_min, u_plus_u_plus, d_plus_d_plus,
                u_plus_u_min, d_plus_d_min, u_min_u_plus, d_min_d_plus,
            )
            O = try
                f(ComplexF64, particle_symmetry, spin_symmetry)
            catch e
                e isa ArgumentError || rethrow()
                continue
            end
            O_sf = f(ComplexF64, particle_symmetry, spin_symmetry; slave_fermion = true)
            @test O_sf ≈ transform_slave_fermion(O)
        end
    end
end

@testset "Hubbard projection" begin
    # the relation to the Hubbard operators: project out the doubly occupied state
    for (particle_symmetry, spin_symmetry) in symmetries
        proj = tj_projector(particle_symmetry, spin_symmetry)
        @test proj isa AbstractTensorMap{Int}
        @test proj * proj' ≈ id(tj_space(particle_symmetry, spin_symmetry))
        for name in (
                :e_num, :h_num, :u_num, :d_num, :S_z, :S_x, :S_plus, :S_min,
                :u_plus_u_min, :d_plus_d_min, :e_plus_e_min, :e_min_e_plus, :e_hopping,
                :u_min_d_min, :d_min_u_min, :u_min_u_min, :d_min_d_min,
                :singlet_plus, :singlet_min, :S_plus_S_min, :S_min_S_plus, :S_exchange,
                :singlet_plus_singlet_min_3site, :singlet_plus_singlet_min_4site,
            )
            O_tj = try
                getproperty(TJOperators, name)(ComplexF64, particle_symmetry, spin_symmetry)
            catch e
                e isa ArgumentError || rethrow()
                continue
            end
            O_hub = getproperty(HO, name)(ComplexF64, particle_symmetry, spin_symmetry)
            projⁿ = reduce(⊗, ntuple(Returns(proj), numout(O_hub)))
            @test projⁿ * O_hub * projⁿ' ≈ O_tj
        end
    end
end

@testset "regression values" begin
    # regression check: hand-written symmetric operators are easily transposed. The dense
    # indices of the basis states are read off from the basis transform, whose columns are
    # ordered as (|0⟩, |↑⟩, |↓⟩) in the plain t-J basis.
    for (P, S) in symmetries
        U = convert(Array, basis_transform(P, S))
        i0 = findfirst(==(1), U[:, 1])
        iu = findfirst(==(1), U[:, 2])

        A = convert(Array, e_plus_e_min(ComplexF64, P, S))
        @test A[iu, i0, i0, iu] ≈ 1        # |↑,0⟩ ↤ |0,↑⟩
        @test abs(A[i0, iu, iu, i0]) < 1.0e-12

        if has_S_z(P, S)
            id_ = findfirst(==(1), U[:, 3])
            B = convert(Array, S_plus_S_min(ComplexF64, P, S))
            @test B[iu, id_, id_, iu] ≈ 1  # |↑,↓⟩ ↤ |↓,↑⟩
            @test abs(B[id_, iu, iu, id_]) < 1.0e-12
        end
    end
end

@testset "basic properties" begin
    for (particle_symmetry, spin_symmetry, slave_fermion) in symmetries_bases
        pspace = tj_space(particle_symmetry, spin_symmetry; slave_fermion)

        # hopping operators
        epem = e_plus_e_min(particle_symmetry, spin_symmetry; slave_fermion)
        emep = e_min_e_plus(particle_symmetry, spin_symmetry; slave_fermion)
        @test epem' ≈ -emep ≈ swap_2sites(epem)
        @test e_hopping(particle_symmetry, spin_symmetry; slave_fermion)' ≈
            e_hopping(particle_symmetry, spin_symmetry; slave_fermion)
        if has_u_num(particle_symmetry, spin_symmetry)
            dpdm = d_plus_d_min(particle_symmetry, spin_symmetry; slave_fermion)
            dmdp = d_min_d_plus(particle_symmetry, spin_symmetry; slave_fermion)
            @test dpdm' ≈ -dmdp ≈ swap_2sites(dpdm)
            upum = u_plus_u_min(particle_symmetry, spin_symmetry; slave_fermion)
            umup = u_min_u_plus(particle_symmetry, spin_symmetry; slave_fermion)
            @test upum' ≈ -umup ≈ swap_2sites(upum)
        else
            @test_throws ArgumentError d_plus_d_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError d_min_d_plus(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError u_plus_u_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError u_min_u_plus(particle_symmetry, spin_symmetry; slave_fermion)
        end

        # number operators
        @test TensorKit.id(pspace) ≈
            h_num(particle_symmetry, spin_symmetry; slave_fermion) +
            e_num(particle_symmetry, spin_symmetry; slave_fermion)
        if has_u_num(particle_symmetry, spin_symmetry)
            nu = u_num(particle_symmetry, spin_symmetry; slave_fermion)
            nd = d_num(particle_symmetry, spin_symmetry; slave_fermion)
            @test e_num(particle_symmetry, spin_symmetry; slave_fermion) ≈ nu + nd
            # no double occupancy
            @test nu * nd ≈ nd * nu ≈ zeros(pspace ← pspace)
        else
            @test_throws ArgumentError u_num(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError d_num(particle_symmetry, spin_symmetry; slave_fermion)
        end

        # singlet operators
        if has_singlet(particle_symmetry, spin_symmetry)
            singm = singlet_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test swap_2sites(singm) ≈ singm
            @test singm ≈ singlet_plus(particle_symmetry, spin_symmetry; slave_fermion)'
            if has_pair(particle_symmetry, spin_symmetry)
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
        end
        if !has_pair(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError u_min_d_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError d_min_u_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError u_plus_d_plus(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError d_plus_u_plus(particle_symmetry, spin_symmetry; slave_fermion)
        end

        # 3-site singlet hopping operator
        O_ijk = Δ⁺ij_Δjk(particle_symmetry, spin_symmetry; slave_fermion)
        O_kji = permute(O_ijk, ((3, 2, 1), (6, 5, 4)))
        @test O_kji ≈ O_ijk'

        # triplet operators
        if has_triplet(particle_symmetry, spin_symmetry)
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

        # spin operators
        if has_spin_ops(particle_symmetry, spin_symmetry)
            test_spin_algebra(
                S_x(particle_symmetry, spin_symmetry; slave_fermion),
                S_y(particle_symmetry, spin_symmetry; slave_fermion),
                S_z(particle_symmetry, spin_symmetry; slave_fermion),
            )
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
        end
        if !has_S_z(particle_symmetry, spin_symmetry)
            @test_throws ArgumentError S_z(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError S_plus_S_min(particle_symmetry, spin_symmetry; slave_fermion)
            @test_throws ArgumentError S_min_S_plus(particle_symmetry, spin_symmetry; slave_fermion)
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

@testset "Exact Diagonalisation" begin
    rng = StableRNG(123)
    t, J = rand(rng, 2)
    true_eigenvals = sort(vcat(-J, fill(-t, 2), fill(t, 2), fill(0.0, 4)))
    for (P, S, slave_fermion) in symmetries_bases
        H = tjhamiltonian(P, S; t, J, mu = 0.0, L = 2, slave_fermion)
        eigenvals = expanded_eigenvalues(H)
        @test eigenvals ≈ true_eigenvals
    end
end
