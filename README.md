<img src="assets/Apoth3osis.webp" alt="Apoth3osis — Formal Mathematics and Verified Software" width="140"/>

<sub><strong>Our tech stack is ontological:</strong><br>
<strong>Hardware — Physics</strong><br>
<strong>Software — Mathematics</strong><br><br>
<strong>Our engineering workflow is simple:</strong> discover, build, grow, learn & teach</sub>

---

<sub>
<strong>Acknowledgment</strong><br>
We humbly thank the collective intelligence of humanity for providing the technology and culture we cherish. We do our best to properly reference the authors of the works utilized herein, though we may occasionally fall short. Our formalization acts as a reciprocal validation—confirming the structural integrity of their original insights while securing the foundation upon which we build. In truth, all creative work is derivative; we stand on the shoulders of those who came before, and our contributions are simply the next link in an unbroken chain of human ingenuity.
</sub>

---

# Abbott Deep Learning Algebra — Finite Lean 4 Formalization and Executable Reference Package

This repository packages a standalone Lean 4, Python, and TypeScript surface for the paper *Weaves, Wires, and Morphisms: Formalizing and Implementing the Algebra of Deep Learning* (Abbott, Zardini, 2026). It focuses on the paper's finite, implementation-facing core: constructed terms, product/remapping structure, direct-sum and flat remapping algebra, axis-stride and array-broadcasted categories, finite Fox-style projection/construction laws, deterministic naturality, a concrete natural-reindexing Yoneda witness, and explicit reference execution for convolution and multi-head attention.

## What Is This?

A researcher package for independently rebuilding the Abbott bridge outside the Heyting monorepo: Lean modules for the formal finite core, plus Python/TypeScript artifacts for explicit lowering and execution.

[![License: Apoth3osis License Stack v1](https://img.shields.io/badge/License-Apoth3osis%20License%20Stack%20v1-blue.svg)](LICENSE.md)

## Repository Layout

- `HeytingLean/Bridge/Abbott/`: Lean 4 formalization of the finite categorical and broadcasting surfaces.
- `HeytingLean/Tests/Bridge/Abbott/`: Lean 4 sanity and regression modules for the Abbott bridge.
- `python/`: executable schema, UID, lowering, and NumPy-backed execution surfaces.
- `ts/`: TypeScript schema mirror for downstream UI or compiler integration.
- `papers/`: source paper PDF.

## Current Scope

- Finite constructed-term system and product-category scaffolding.
- Finite remapping algebra with direct sums, segmented flattening, and flat remappings.
- Finite remapping-based Fox projection/construction laws.
- Deterministic naturality over finite remappings.
- Concrete natural-reindexing Yoneda sliding over explicit `Reindexing` witnesses.
- Explicit affine reindexing plus NumPy-backed execution for:
  - convolution with explicit window extraction
  - multi-head self-attention with explicit head splitting

## Honest Boundaries

- This package does **not** claim the full general categorical theory of the paper.
- The Fox/Yoneda layer here is a finite reference formalization, not the full abstract theorem family.
- The natural-remapping branch is tied to concrete `Reindexing` witnesses built from `StrideMor.purePermutation`; it is not yet promoted to every possible broadcasted operation structure in the paper.
- Torch execution is optional. When `torch` is absent, execution runs through the NumPy reference backend.
- No attempt is made here to treat implicit backend broadcasting as the semantic source of truth.

## Lean Modules

Primary bridge surface:

- `HeytingLean.Bridge.Abbott`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Syntax`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.ProductCategory`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Elemental`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Remapping`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.AxisStride`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.ArrayBroadcasted`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Broadcasting`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Laws`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples.Convolution`
- `HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples.Attention`

Test surface:

- `HeytingLean.Tests.Bridge.Abbott`
- `HeytingLean.Tests.Bridge.Abbott.DeepLearningAlgebraSanity`
- `HeytingLean.Tests.Bridge.Abbott.FoxSanity`
- `HeytingLean.Tests.Bridge.Abbott.BroadcastingSanity`
- `HeytingLean.Tests.Bridge.Abbott.ConvolutionSanity`

## Theorem Inventory

The finite law layer currently exposes:

- `FoundationalLaws.deterministicNaturality`
- `FoundationalLaws.foxFreeConstruction_projection`
- `FoundationalLaws.foxUniqueIdentification`
- `FoundationalLaws.yonedaSliding_ofNatural`
- `FoundationalLaws.yonedaSliding_ofDeterministic`
- `FoundationalLaws.NaturalReindexing.rankCompatible`
- `FoundationalLaws.NaturalReindexing.yonedaSliding`
- `FoundationalLaws.NaturalReindexing.yonedaSliding_ofDeterministic`

The remapping layer additionally exposes the Appendix A.2 finite constructions:

- `FiniteRemapping.directSum`
- `FiniteRemapping.joinSegments`
- `FiniteRemapping.splitSegments`
- `FiniteRemapping.directSumFamily`
- `FiniteRemapping.pullSegmentSizes`
- `FiniteRemapping.flatRemapping`

## Build and Verify

Lean:

```bash
lake build HeytingLean.Bridge.Abbott
lake build HeytingLean.Tests.Bridge.Abbott
```

Python:

```bash
python3 -m py_compile python/*.py
python3 - <<'PY'
import sys
sys.path.insert(0, "python")
from examples import (
    convolution_example,
    convolution_sample_tensors,
    attention_example,
    attention_sample_tensors,
)
from torch_lowering import lower_to_torch_plan, maybe_execute_torch

conv_schema = convolution_example()
conv_plan = lower_to_torch_plan(conv_schema)
conv = maybe_execute_torch(conv_plan, list(convolution_sample_tensors()), conv_schema)
print(conv["backend"], conv["result"].shape)

attn_schema = attention_example()
attn_plan = lower_to_torch_plan(attn_schema)
attn = maybe_execute_torch(attn_plan, list(attention_sample_tensors()), attn_schema)
print(attn["backend"], attn["result"].shape)
PY
```

## Provenance

- Paper source: `papers/abbott-zardini-weaves-wires-morphisms-2026.pdf`
- Monorepo source: the Abbott bridge and companion project were extracted from HeytingLean for standalone verification.
- Package intent: make the finite formalization and reference execution surfaces independently inspectable.

## License

[Apoth3osis License Stack v1](LICENSE.md)
