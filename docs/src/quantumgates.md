```@meta
CollapsedDocStrings = true
CurrentModule = TensorKitTensors.QuantumGates
```

# Quantum gates

Common single-, two- and three-qubit gates used in quantum circuits, returned as `TensorMap`s on qubit spaces.

## Conventions

### Basis ordering

A qubit lives on a two-dimensional space with computational basis ordered as

```math
|0⟩,\; |1⟩ \quad \text{(row/column 1 = } |0⟩\text{)}
```

Multi-qubit gates act on ``V^{⊗n}`` with the first qubit as the leftmost tensor factor, so a controlled gate uses the *first* qubit(s) as control and the *last* as target.

### Symmetry sectors

| Symmetry | Physical meaning | Sector label | Representable gates |
|---|---|---|---|
| `Trivial` | none | — | all gates, full ``2^n \times 2^n`` matrix |
| `U1Irrep` | excitation-number conservation | charge ``\in \{0, 1\}`` per qubit | only gates that conserve the number of ``|1\rangle``'s |

!!! note "U(1) charge = excitation number"
    Under `U1Irrep` the two sectors of a qubit carry charge ``0`` (``|0\rangle``) and ``1`` (``|1\rangle``), i.e. the conserved quantity is the number of excited qubits.
    Only number-conserving gates are representable in this symmetry: [`pauli_z`](@ref), [`proj_0`](@ref), [`proj_1`](@ref), [`s_gate`](@ref), [`t_gate`](@ref), [`phase_shift`](@ref), [`rotation_z`](@ref), [`cz`](@ref), [`cs`](@ref), [`cphase`](@ref), [`swap`](@ref), [`iswap`](@ref), [`rotation_zz`](@ref) and [`fredkin`](@ref).
    Gates that flip qubits (X, Y, H, CNOT, Toffoli, …) throw an `ArgumentError` when requested with `U1Irrep`.

## Gate overview

| Function | Alias(es) | Sites | Supported symmetries |
|---|---|---|---|
| [`qubit_space`](@ref) | — | — | `Trivial`, `U1Irrep` |
| [`pauli_x`](@ref) | `X` | 1 | `Trivial` |
| [`pauli_y`](@ref) | `Y` | 1 | `Trivial` |
| [`pauli_z`](@ref) | `Z` | 1 | `Trivial`, `U1Irrep` |
| [`proj_0`](@ref) | `P0` | 1 | `Trivial`, `U1Irrep` |
| [`proj_1`](@ref) | `P1` | 1 | `Trivial`, `U1Irrep` |
| [`hadamard`](@ref) | `H` | 1 | `Trivial` |
| [`s_gate`](@ref) | `S` | 1 | `Trivial`, `U1Irrep` |
| [`t_gate`](@ref) | `T` | 1 | `Trivial`, `U1Irrep` |
| [`phase_shift`](@ref) | `P` | 1 | `Trivial`, `U1Irrep` |
| [`rotation_x`](@ref) | `Rx` | 1 | `Trivial` |
| [`rotation_y`](@ref) | `Ry` | 1 | `Trivial` |
| [`rotation_z`](@ref) | `Rz` | 1 | `Trivial`, `U1Irrep` |
| [`cnot`](@ref) | `CNOT`, `CX` | 2 | `Trivial` |
| [`cy`](@ref) | `CY` | 2 | `Trivial` |
| [`cz`](@ref) | `CZ` | 2 | `Trivial`, `U1Irrep` |
| [`ch`](@ref) | `CH` | 2 | `Trivial` |
| [`cs`](@ref) | `CS` | 2 | `Trivial`, `U1Irrep` |
| [`cphase`](@ref) | `CP` | 2 | `Trivial`, `U1Irrep` |
| [`swap`](@ref) | `SWAP` | 2 | `Trivial`, `U1Irrep` |
| [`iswap`](@ref) | `ISWAP` | 2 | `Trivial`, `U1Irrep` |
| [`dcx`](@ref) | `DCX` | 2 | `Trivial` |
| [`ecr`](@ref) | `ECR` | 2 | `Trivial` |
| [`rotation_xx`](@ref) | `Rxx` | 2 | `Trivial` |
| [`rotation_yy`](@ref) | `Ryy` | 2 | `Trivial` |
| [`rotation_zz`](@ref) | `Rzz` | 2 | `Trivial`, `U1Irrep` |
| [`rotation_zx`](@ref) | `Rzx` | 2 | `Trivial` |
| [`toffoli`](@ref) | `TOFFOLI`, `CCX` | 3 | `Trivial` |
| [`fredkin`](@ref) | `FREDKIN`, `CSWAP` | 3 | `Trivial`, `U1Irrep` |

The adjoint gates ``S^\dagger`` and ``T^\dagger`` are obtained as `s_gate()'` and `t_gate()'`.

## API

```@autodocs
Modules = [QuantumGates]
```
