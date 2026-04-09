# Abbott Deep Learning Algebra

This repository packages a Lean 4 categorical bridge formalization and executable reference
surface for Vincent Abbott and Gioele Zardini's paper *Weaves, Wires, and
Morphisms: Formalizing and Implementing the Algebra of Deep Learning*.

It includes the categorical bridge modules, categorical Fox/Yoneda theorem
surfaces, and categorical example refinements, while retaining the finite witness
layer as the specialization and executable regression surface.

## What This Repository Contains

- Lean modules for:
  - constructed terms
  - product-category wiring
  - remapping algebra
  - axis-stride and array-broadcasted structures
  - categorical product / stride / array / broadcast layers
  - categorical interpretation functors
  - categorical Fox and Yoneda theorem surfaces
  - categorical convolution and attention refinements
  - finite Fox-style laws and concrete natural-reindexing witnesses
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
- categorical product, stride, array, and broadcast categories
- categorical interpretation layer into product and broadcast semantics
- categorical Fox projection/free-construction theorems in the semantic `Type` category
- categorical Yoneda sliding theorems in the semantic `Type` category
- categorical convolution and attention example refinements back to the finite layer
- finite Fox projection/free-construction laws
- deterministic naturality
- concrete `NaturalReindexing` witnesses
- finite broadcast-layer Yoneda witness in `Yoneda.lean`
- NumPy-backed reference execution for convolution and attention

## Honest Boundaries

- The Fox/Yoneda categorical theorem files are stated in the semantic category `Type`
  over finite tuple objects. They are not yet stated as equalities internal to the
  `BroadcastCat` path category itself.
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
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Interpretation`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Fox`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Yoneda`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples.Convolution`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples.Attention`
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
