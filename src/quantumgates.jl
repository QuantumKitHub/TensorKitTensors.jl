module QuantumGates

using TensorKit
using LinearAlgebra: I
import ..TensorKitTensors: symmetrize, desymmetrize, @operator

export qubit_space
export pauli_x, pauli_y, pauli_z, proj_0, proj_1, hadamard, s_gate, t_gate
export phase_shift, rotation_x, rotation_y, rotation_z
export cnot, cy, cz, cphase, ch, cs, swap, iswap, dcx, ecr
export rotation_xx, rotation_yy, rotation_zz, rotation_zx
export toffoli, fredkin
export X, Y, Z, P0, P1, H, S, T, P, Rx, Ry, Rz
export CNOT, CX, CY, CZ, CP, CH, CS, SWAP, ISWAP, DCX, ECR
export Rxx, Ryy, Rzz, Rzx
export TOFFOLI, CCX, FREDKIN, CSWAP

"""
    qubit_space([symmetry::Type{<:Sector}])

Return the local Hilbert space of a single qubit with basis ``|0\\rangle, |1\\rangle``.

| Symmetry | Space |
|---|---|
| `Trivial` | `ComplexSpace(2)` |
| `U1Irrep` | `U1Space(0 => 1, 1 => 1)` (charge = number of excitations) |
"""
qubit_space(::Type{Trivial} = Trivial) = ComplexSpace(2)
qubit_space(::Type{U1Irrep}) = U1Space(0 => 1, 1 => 1)
qubit_space(S::Type{<:Sector}) = throw(ArgumentError("invalid symmetry `$S`"))

# Basis transformation from the trivial |0⟩, |1⟩ basis to the symmetric basis. The
# computational basis already coincides with the U(1) charge eigenbasis (charge = number of
# excitations), so both transformations are the identity.
basis_transform(::Type{Trivial}) =
    TensorMap(Matrix{Int}(I, 2, 2), qubit_space(Trivial) ← qubit_space(Trivial))
basis_transform(::Type{U1Irrep}) =
    TensorMap(Matrix{Int}(I, 2, 2), desymmetrize(qubit_space(U1Irrep)) ← qubit_space(Trivial))
basis_transform(S::Type{<:Sector}) = throw(ArgumentError("invalid symmetry `$S`"))

# Symmetrize a gate through its basis transformation. `symmetrize` throws an `ArgumentError`
# when the gate does not respect the requested symmetry, so gates that break excitation-number
# conservation automatically reject `U1Irrep`. The qubit space does not depend on any gate
# parameters, so forwarded keyword arguments (e.g. `θ`) are dropped here.
_symmetrize_operator(O::AbstractTensorMap, symmetry::Type{<:Sector}; kwargs...) =
    symmetrize(O, basis_transform(symmetry), qubit_space(symmetry))

# Pauli gates
# -----------
"""
    pauli_x([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    X([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])

The Pauli-X (NOT) gate ``\\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}``.

Supported symmetries: `Trivial`.
"""
@operator X function pauli_x(elt::Type{<:Number}, ::Type{Trivial})
    return TensorMap(elt[0 1; 1 0], qubit_space() ← qubit_space())
end

"""
    pauli_y([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])
    Y([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])

The Pauli-Y gate ``\\begin{pmatrix} 0 & -i \\\\ i & 0 \\end{pmatrix}``.

Supported symmetries: `Trivial`.
"""
@operator Y function pauli_y(elt::Type{<:Complex}, ::Type{Trivial})
    return TensorMap(elt[0 -im; im 0], qubit_space() ← qubit_space())
end

"""
    pauli_z([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    Z([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])

The Pauli-Z gate ``\\begin{pmatrix} 1 & 0 \\\\ 0 & -1 \\end{pmatrix}``.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator Z function pauli_z(elt::Type{<:Number}, ::Type{Trivial})
    return TensorMap(elt[1 0; 0 -1], qubit_space() ← qubit_space())
end

# Projectors
# ----------
"""
    proj_0([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    P0([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])

The projector onto ``|0\\rangle``, ``|0\\rangle\\langle 0| = \\tfrac{1}{2}(I + Z)``.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator P0 function proj_0(elt::Type{<:Number}, ::Type{Trivial})
    Z = pauli_z(elt, Trivial)
    return (one(Z) + Z) / 2
end

"""
    proj_1([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    P1([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])

The projector onto ``|1\\rangle``, ``|1\\rangle\\langle 1| = \\tfrac{1}{2}(I - Z)``.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator P1 function proj_1(elt::Type{<:Number}, ::Type{Trivial})
    Z = pauli_z(elt, Trivial)
    return (one(Z) - Z) / 2
end

# Derived single-qubit gates
# --------------------------
"""
    hadamard([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    H([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])

The Hadamard gate ``H = \\tfrac{1}{\\sqrt 2}(X + Z)``.

Supported symmetries: `Trivial`.
"""
@operator H function hadamard(elt::Type{<:Number}, ::Type{Trivial})
    return (pauli_x(elt, Trivial) + pauli_z(elt, Trivial)) / sqrt(2)
end

"""
    phase_shift([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)
    P([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)

The phase-shift gate ``\\mathrm{diag}(1, e^{i\\theta}) = P_0 + e^{i\\theta} P_1``. The rotation
angle `θ` is a required keyword; the ASCII name `theta` is accepted as an alias.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator P function phase_shift(elt::Type{<:Complex}, ::Type{Trivial}; θ = nothing, theta = θ)
    α = @something theta throw(ArgumentError("phase angle `θ` (or `theta`) is required"))
    return proj_0(elt, Trivial) + cis(α) * proj_1(elt, Trivial)
end

"""
    s_gate([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])
    S([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])

The phase gate ``S = \\sqrt Z = \\mathrm{diag}(1, i)``, i.e. `phase_shift(; θ=π/2)`. Its adjoint ``S^\\dagger`` is `s_gate()'`.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator S function s_gate(elt::Type{<:Complex}, ::Type{Trivial})
    return phase_shift(elt, Trivial; θ = π / 2)
end

"""
    t_gate([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])
    T([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])

The ``T`` (π/8) gate ``T = \\sqrt S = \\mathrm{diag}(1, e^{i\\pi/4})``, i.e. `phase_shift(; θ=π/4)`. Its adjoint ``T^\\dagger`` is `t_gate()'`.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator T function t_gate(elt::Type{<:Complex}, ::Type{Trivial})
    return phase_shift(elt, Trivial; θ = π / 4)
end

"""
    rotation_x([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)
    Rx([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)

The x-rotation gate ``e^{-i\\theta X/2} = \\cos\\tfrac{\\theta}{2}\\,I - i\\sin\\tfrac{\\theta}{2}\\,X``. The rotation
angle `θ` is a required keyword; the ASCII name `theta` is accepted as an alias.

Supported symmetries: `Trivial`.
"""
@operator Rx function rotation_x(elt::Type{<:Complex}, ::Type{Trivial}; θ = nothing, theta = θ)
    α = @something theta throw(ArgumentError("rotation angle `θ` (or `theta`) is required"))
    x = pauli_x(elt, Trivial)
    return cos(α / 2) * one(x) - im * sin(α / 2) * x
end

"""
    rotation_y([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)
    Ry([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)

The y-rotation gate ``e^{-i\\theta Y/2} = \\cos\\tfrac{\\theta}{2}\\,I - i\\sin\\tfrac{\\theta}{2}\\,Y``. The rotation
angle `θ` is a required keyword; the ASCII name `theta` is accepted as an alias.

Supported symmetries: `Trivial`.
"""
@operator Ry function rotation_y(elt::Type{<:Complex}, ::Type{Trivial}; θ = nothing, theta = θ)
    α = @something theta throw(ArgumentError("rotation angle `θ` (or `theta`) is required"))
    y = pauli_y(elt, Trivial)
    return cos(α / 2) * one(y) - im * sin(α / 2) * y
end

"""
    rotation_z([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)
    Rz([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)

The z-rotation gate ``e^{-i\\theta Z/2} = \\cos\\tfrac{\\theta}{2}\\,I - i\\sin\\tfrac{\\theta}{2}\\,Z``. The rotation
angle `θ` is a required keyword; the ASCII name `theta` is accepted as an alias.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator Rz function rotation_z(elt::Type{<:Complex}, ::Type{Trivial}; θ = nothing, theta = θ)
    α = @something theta throw(ArgumentError("rotation angle `θ` (or `theta`) is required"))
    z = pauli_z(elt, Trivial)
    return cos(α / 2) * one(z) - im * sin(α / 2) * z
end

# Two-qubit gates
# ---------------
"""
    cnot([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    CNOT([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    CX([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])

The controlled-NOT gate ``P_0 \\otimes I + P_1 \\otimes X``: applies [`pauli_x`](@ref) to the second qubit if the first is ``|1\\rangle``.

Supported symmetries: `Trivial`.
"""
@operator CNOT function cnot(elt::Type{<:Number}, ::Type{Trivial})
    q = qubit_space()
    return proj_0(elt, Trivial) ⊗ id(q) + proj_1(elt, Trivial) ⊗ pauli_x(elt, Trivial)
end
const CX = cnot

"""
    cy([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])
    CY([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])

The controlled-Y gate ``P_0 \\otimes I + P_1 \\otimes Y``: applies [`pauli_y`](@ref) to the second qubit if the first is ``|1\\rangle``.

Supported symmetries: `Trivial`.
"""
@operator CY function cy(elt::Type{<:Complex}, ::Type{Trivial})
    q = qubit_space()
    return proj_0(elt, Trivial) ⊗ id(q) + proj_1(elt, Trivial) ⊗ pauli_y(elt, Trivial)
end

"""
    ch([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    CH([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])

The controlled-Hadamard gate ``P_0 \\otimes I + P_1 \\otimes H``: applies [`hadamard`](@ref) to the second qubit if the first is ``|1\\rangle``.

Supported symmetries: `Trivial`.
"""
@operator CH function ch(elt::Type{<:Number}, ::Type{Trivial})
    q = qubit_space()
    return proj_0(elt, Trivial) ⊗ id(q) + proj_1(elt, Trivial) ⊗ hadamard(elt, Trivial)
end

"""
    cphase([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)
    CP([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)

The controlled phase-shift gate ``\\mathrm{diag}(1, 1, 1, e^{i\\theta}) = P_0 \\otimes I + P_1 \\otimes \\mathrm{phase\\_shift}(\\theta)``.
The rotation angle `θ` is a required keyword; the ASCII name `theta` is accepted as an alias.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator CP function cphase(elt::Type{<:Complex}, ::Type{Trivial}; θ = nothing, theta = θ)
    α = @something theta throw(ArgumentError("phase angle `θ` (or `theta`) is required"))
    q = qubit_space()
    return proj_0(elt, Trivial) ⊗ id(q) + proj_1(elt, Trivial) ⊗ phase_shift(elt, Trivial; θ = α)
end

"""
    cz([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])
    CZ([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])

The controlled-Z gate ``\\mathrm{diag}(1, 1, 1, -1)``, i.e. `cphase(; θ=π)`.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator CZ function cz(elt::Type{<:Complex}, ::Type{Trivial})
    return cphase(elt, Trivial; θ = π)
end

"""
    cs([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])
    CS([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])

The controlled-``S`` gate ``\\mathrm{diag}(1, 1, 1, i)``, i.e. `cphase(; θ=π/2)`.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator CS function cs(elt::Type{<:Complex}, ::Type{Trivial})
    return cphase(elt, Trivial; θ = π / 2)
end

"""
    swap([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    SWAP([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])

The gate that swaps two qubits, ``|ab\\rangle \\mapsto |ba\\rangle``.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator SWAP function swap(elt::Type{<:Number}, ::Type{Trivial})
    q = qubit_space()
    return one(elt) * permute(id(q ⊗ q), ((1, 2), (4, 3)))
end

"""
    iswap([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])
    ISWAP([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])

The iSWAP gate, swapping two qubits with an extra factor ``i`` on the ``|01\\rangle \\leftrightarrow |10\\rangle`` amplitudes.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator ISWAP function iswap(elt::Type{<:Complex}, ::Type{Trivial})
    D = proj_0(elt, Trivial) ⊗ proj_0(elt, Trivial) + proj_1(elt, Trivial) ⊗ proj_1(elt, Trivial)
    return D + im * (swap(elt, Trivial) - D)
end

"""
    dcx([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    DCX([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])

The double-CNOT gate: two back-to-back CNOTs with alternating control and target.

Supported symmetries: `Trivial`.
"""
@operator DCX function dcx(elt::Type{<:Number}, ::Type{Trivial})
    q = qubit_space()
    cx = id(q) ⊗ proj_0(elt, Trivial) + pauli_x(elt, Trivial) ⊗ proj_1(elt, Trivial)
    return cx * cnot(elt, Trivial)
end

"""
    ecr([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])
    ECR([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}])

The echoed cross-resonance gate ``\\mathrm{ECR} = \\tfrac{1}{\\sqrt 2}(I \\otimes X - X \\otimes Y)``.

Supported symmetries: `Trivial`.
"""
@operator ECR function ecr(elt::Type{<:Complex}, ::Type{Trivial})
    q = qubit_space()
    x, y = pauli_x(elt, Trivial), pauli_y(elt, Trivial)
    return (id(q) ⊗ x - x ⊗ y) / sqrt(2)
end

# Two-qubit rotation gates
# ------------------------
"""
    rotation_xx([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)
    Rxx([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)

The XX-rotation (Ising coupling) gate ``e^{-i\\theta\\, X \\otimes X / 2} = \\cos\\tfrac{\\theta}{2}\\,I \\otimes I - i\\sin\\tfrac{\\theta}{2}\\,X \\otimes X``.
The rotation angle `θ` is a required keyword; the ASCII name `theta` is accepted as an alias.

Supported symmetries: `Trivial`.
"""
@operator Rxx function rotation_xx(elt::Type{<:Complex}, ::Type{Trivial}; θ = nothing, theta = θ)
    α = @something theta throw(ArgumentError("rotation angle `θ` (or `theta`) is required"))
    xx = pauli_x(elt, Trivial) ⊗ pauli_x(elt, Trivial)
    return cos(α / 2) * one(xx) - im * sin(α / 2) * xx
end

"""
    rotation_yy([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)
    Ryy([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)

The YY-rotation gate ``e^{-i\\theta\\, Y \\otimes Y / 2} = \\cos\\tfrac{\\theta}{2}\\,I \\otimes I - i\\sin\\tfrac{\\theta}{2}\\,Y \\otimes Y``.
The rotation angle `θ` is a required keyword; the ASCII name `theta` is accepted as an alias.

Supported symmetries: `Trivial`.
"""
@operator Ryy function rotation_yy(elt::Type{<:Complex}, ::Type{Trivial}; θ = nothing, theta = θ)
    α = @something theta throw(ArgumentError("rotation angle `θ` (or `theta`) is required"))
    yy = pauli_y(elt, Trivial) ⊗ pauli_y(elt, Trivial)
    return cos(α / 2) * one(yy) - im * sin(α / 2) * yy
end

"""
    rotation_zz([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)
    Rzz([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)

The ZZ-rotation gate ``e^{-i\\theta\\, Z \\otimes Z / 2} = \\cos\\tfrac{\\theta}{2}\\,I \\otimes I - i\\sin\\tfrac{\\theta}{2}\\,Z \\otimes Z``.
The rotation angle `θ` is a required keyword; the ASCII name `theta` is accepted as an alias.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator Rzz function rotation_zz(elt::Type{<:Complex}, ::Type{Trivial}; θ = nothing, theta = θ)
    α = @something theta throw(ArgumentError("rotation angle `θ` (or `theta`) is required"))
    zz = pauli_z(elt, Trivial) ⊗ pauli_z(elt, Trivial)
    return cos(α / 2) * one(zz) - im * sin(α / 2) * zz
end

"""
    rotation_zx([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)
    Rzx([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; θ)

The ZX-rotation gate ``e^{-i\\theta\\, Z \\otimes X / 2} = \\cos\\tfrac{\\theta}{2}\\,I \\otimes I - i\\sin\\tfrac{\\theta}{2}\\,Z \\otimes X``.
The rotation angle `θ` is a required keyword; the ASCII name `theta` is accepted as an alias.

Supported symmetries: `Trivial`.
"""
@operator Rzx function rotation_zx(elt::Type{<:Complex}, ::Type{Trivial}; θ = nothing, theta = θ)
    α = @something theta throw(ArgumentError("rotation angle `θ` (or `theta`) is required"))
    zx = pauli_z(elt, Trivial) ⊗ pauli_x(elt, Trivial)
    return cos(α / 2) * one(zx) - im * sin(α / 2) * zx
end

# Three-qubit gates
# -----------------
"""
    toffoli([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    TOFFOLI([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    CCX([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])

The Toffoli (CCX) gate ``P_0 \\otimes I \\otimes I + P_1 \\otimes \\mathrm{cnot}``: applies [`pauli_x`](@ref) to the third qubit if the first two are both ``|1\\rangle``.

Supported symmetries: `Trivial`.
"""
@operator TOFFOLI function toffoli(elt::Type{<:Number}, ::Type{Trivial})
    q = qubit_space()
    return proj_0(elt, Trivial) ⊗ id(q) ⊗ id(q) + proj_1(elt, Trivial) ⊗ cnot(elt, Trivial)
end
const CCX = toffoli

"""
    fredkin([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    FREDKIN([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])
    CSWAP([eltype::Type{<:Number}], [symmetry::Type{<:Sector}])

The Fredkin (CSWAP) gate ``P_0 \\otimes I \\otimes I + P_1 \\otimes \\mathrm{swap}``: swaps the last two qubits if the first is ``|1\\rangle``.

Supported symmetries: `Trivial`, `U1Irrep`.
"""
@operator FREDKIN function fredkin(elt::Type{<:Number}, ::Type{Trivial})
    q = qubit_space()
    return proj_0(elt, Trivial) ⊗ id(q) ⊗ id(q) + proj_1(elt, Trivial) ⊗ swap(elt, Trivial)
end
const CSWAP = fredkin

end
