```@meta
CurrentModule = TensorKitTensors.TJOperators
CollapsedDocStrings = true
```

# TJ operators

## Relation to the Hubbard model

The t-J model is the Hubbard model restricted to the subspace without double occupancy. The
operators of this module are defined accordingly: each of them is the projection of the
[Hubbard operator](hubbardoperators.md) of the same name onto the t-J space, through the
``3 ← 4`` isometry [`tj_projector`](@ref). The only Hubbard operators without a t-J
counterpart are the double-occupancy operators `ud_num` and `half_ud_num`, which project to
zero.

```@autodocs
Modules = [TJOperators]
```
