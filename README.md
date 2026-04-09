# Abbott Deep Learning Algebra

This repository packages a finite Lean 4 formalization and executable reference
surface for Vincent Abbott and Gioele Zardini's paper *Weaves, Wires, and
Morphisms: Formalizing and Implementing the Algebra of Deep Learning*.

It is a finite witness package, not a claim of having completed the paper's full
abstract categorical theory.

## What This Repository Contains

- Lean modules for:
  - constructed terms
  - product-category wiring
  - remapping algebra
  - axis-stride and array-broadcasted structures
  - finite Fox-style laws
  - concrete natural-reindexing and broadcast-level Yoneda witnesses
- Python reference execution for:
  - convolution
  - self-attention
- TypeScript schema mirrors for downstream UI and artifact inspection

## Repository Layout

- `HeytingLean/Bridge/Abbott/`
  - finite Lean surface for the Abbott bridge
- `HeytingLean/Tests/Bridge/Abbott/`
  - Lean sanity and regression modules
- `python/`
  - executable schema, lowering, and NumPy reference execution
- `ts/`
  - TypeScript schema mirror
- `papers/`
  - source paper PDF

## Current Scope

- finite constructed-term system and product scaffold
- Appendix A.2 finite remapping algebra:
  - `directSum`
  - `joinSegments`
  - `splitSegments`
  - `directSumFamily`
  - `pullSegmentSizes`
  - `flatRemapping`
  - theorem-level roundtrip/coherence witnesses
- explicit axis-stride and array-broadcasted structures
- finite Fox projection/free-construction laws
- deterministic naturality
- concrete `NaturalReindexing` witnesses
- finite broadcast-layer Yoneda witness in `Yoneda.lean`
- NumPy-backed reference execution for convolution and attention

## Honest Boundaries

- This does not claim the paper's full abstract categorical theory.
- The Yoneda result here is a finite broadcast-witness surface, not the full
  general theorem family as an abstract category package.
- The executor is a NumPy reference runtime.
- When `torch` is installed, results may be wrapped as torch tensors, but this
  repository does not yet claim native torch kernel execution.
- Implicit backend broadcasting is not treated as the semantic source of truth.

## Schema Convention

Executable reindexings use:

- `dom_axes`
- `cod_axes`
- `linear`
- `offset`

This matches the Lean pullback orientation used by `StrideMor`: rows correspond
to domain axes, columns correspond to codomain axes, and the affine map computes
domain coordinates from codomain coordinates.

## Key Lean Modules

- `HeytingLean.Bridge.Abbott`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Remapping`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.AxisStride`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Broadcasting`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Laws`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Yoneda`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples.Convolution`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples.Attention`

## Verification

Lean:

```bash
lake build HeytingLean.Bridge.Abbott
lake build HeytingLean.Tests.Bridge.Abbott
```

Python:

```bash
python3 -m py_compile python/*.py
cd python
python3 - <<'PY'
import json
from examples import (
    attention_example,
    attention_sample_tensors,
    convolution_example,
    convolution_sample_tensors,
)
from torch_lowering import execute_reference_backend, lower_to_torch_plan

conv_schema = convolution_example()
conv = execute_reference_backend(lower_to_torch_plan(conv_schema), list(convolution_sample_tensors()), conv_schema)

attn_schema = attention_example()
attn = execute_reference_backend(lower_to_torch_plan(attn_schema), list(attention_sample_tensors()), attn_schema)

print(json.dumps({
    "conv_backend": conv["backend"],
    "conv_shape": list(conv["result"].shape),
    "attn_backend": attn["backend"],
    "attn_shape": list(attn["result"].shape),
}, sort_keys=True))
PY
```

## Provenance

- paper source: `papers/abbott-zardini-weaves-wires-morphisms-2026.pdf`
- source-of-truth development repo: `heyting-imm`
- public page: the corresponding PPC page in `apoth3osis_webapp`

## License

See `LICENSE.md`.
