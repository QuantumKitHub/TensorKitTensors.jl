using TensorKit
using LinearAlgebra: eigvals
using Test
include("testsetup.jl")
using .TensorKitTensorsTestSetup
using TensorKitTensors.HubbardOperators
using StableRNGs

all_symmetries = [
    (Trivial, Trivial), (Trivial, U1Irrep), (Trivial, SU2Irrep),
    (U1Irrep, Trivial), (U1Irrep, U1Irrep), (U1Irrep, SU2Irrep),
    (SU2Irrep, Trivial), (SU2Irrep, U1Irrep), (SU2Irrep, SU2Irrep),
]

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
    for (P, S) in all_symmetries
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
    for (particle_symmetry, spin_symmetry) in all_symmetries
        space = @inferred hubbard_space(particle_symmetry, spin_symmetry)
        @test dim(space) == 4

        # compatible with all symmetry combinations
        for f in (half_ud_num, e_hopping, S_exchange)
            O = f(ComplexF64, particle_symmetry, spin_symmetry)
            O_triv = f(ComplexF64, Trivial, Trivial)
            test_operator(O, O_triv)
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
                    test_operator(O, O_triv)
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
    for (P, S) in all_symmetries
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
    for (particle_symmetry, spin_symmetry) in all_symmetries
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
            if !has_S_z(particle_symmetry, spin_symmetry)
                @test_throws ArgumentError S_z(particle_symmetry, spin_symmetry)
            end
        end
    end
end

function hubbard_hamiltonian(particle_symmetry, spin_symmetry; t, U, mu)
    L = length(t) + 1
    @assert length(t) + 1 == length(U) == length(mu)
    hopping = e_hopping(particle_symmetry, spin_symmetry)
    interaction = ud_num(particle_symmetry, spin_symmetry)
    chemical_potential = e_num(particle_symmetry, spin_symmetry)
    I = id(hubbard_space(particle_symmetry, spin_symmetry))
    H = sum(1:(L - 1)) do i
        return reduce(⊗, insert!(collect(Any, fill(I, L - 2)), i, hopping * -t[i]))
    end +
        sum(1:L) do i
        return reduce(⊗, insert!(collect(Any, fill(I, L - 1)), i, interaction * U[i]))
    end +
        sum(1:L) do i
        return reduce(⊗, insert!(collect(Any, fill(I, L - 1)), i, chemical_potential * -mu[i]))
    end
    return H
end
# particle-hole symmetric Hamiltonian (mu = U/2), compatible with all symmetries
function hubbard_hamiltonian_ph(particle_symmetry, spin_symmetry; t, U)
    L = length(t) + 1
    @assert length(t) + 1 == length(U)
    hopping = e_hopping(particle_symmetry, spin_symmetry)
    interaction = half_ud_num(particle_symmetry, spin_symmetry)
    I = id(hubbard_space(particle_symmetry, spin_symmetry))
    H = sum(1:(L - 1)) do i
        return reduce(⊗, insert!(collect(Any, fill(I, L - 2)), i, hopping * -t[i]))
    end + sum(1:L) do i
        return reduce(⊗, insert!(collect(Any, fill(I, L - 1)), i, interaction * U[i]))
    end
    return H
end

@testset "spectrum" begin
    L = 4
    t = randn(L - 1)
    U = randn(L)
    mu = randn(L)

    H_triv = hubbard_hamiltonian(Trivial, Trivial; t, U, mu)
    vals_triv = mapreduce(vcat, pairs(eigvals(H_triv))) do (c, v)
        return repeat(real.(v), dim(c))
    end
    sort!(vals_triv)

    for (particle_symmetry, spin_symmetry) in all_symmetries
        has_e_num(particle_symmetry, spin_symmetry) || continue
        particle_symmetry == spin_symmetry == Trivial && continue
        H_symm = hubbard_hamiltonian(particle_symmetry, spin_symmetry; t, U, mu)
        vals_symm = mapreduce(vcat, pairs(eigvals(H_symm))) do (c, v)
            return repeat(real.(v), dim(c))
        end
        sort!(vals_symm)
        @test vals_triv ≈ vals_symm
    end

    # particle-hole symmetric spectrum: compatible with all symmetry combinations
    H_triv = hubbard_hamiltonian_ph(Trivial, Trivial; t, U)
    vals_triv = mapreduce(vcat, pairs(eigvals(H_triv))) do (c, v)
        return repeat(real.(v), dim(c))
    end
    sort!(vals_triv)

    for (particle_symmetry, spin_symmetry) in all_symmetries
        particle_symmetry == spin_symmetry == Trivial && continue
        H_symm = hubbard_hamiltonian_ph(particle_symmetry, spin_symmetry; t, U)
        vals_symm = mapreduce(vcat, pairs(eigvals(H_symm))) do (c, v)
            return repeat(real.(v), dim(c))
        end
        sort!(vals_symm)
        @test vals_triv ≈ vals_symm
    end
end

@testset "Exact diagonalisation" begin
    for (particle_symmetry, spin_symmetry) in all_symmetries
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
            vcat(
                repeat([-t], 2), [E⁻], repeat([0], 4),
                repeat([t], 2),
                repeat([U - t], 2), [U], [E⁺], repeat([U + t], 2),
                [2 * U]
            )
        )
        eigenvals = expanded_eigenvalues(H_triv; L)
        @test eigenvals ≈ true_eigenvals
    end
end
