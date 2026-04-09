import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Syntax

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra

/-- A product object is an explicit ordered family of factors. -/
structure ProductObject where
  factors : Array ObjectTerm
  deriving Repr

namespace ProductObject

def empty : ProductObject :=
  { factors := #[] }

def singleton (obj : ObjectTerm) : ProductObject :=
  { factors := #[obj] }

def append (lhs rhs : ProductObject) : ProductObject :=
  { factors := lhs.factors ++ rhs.factors }

def factorCount (obj : ProductObject) : Nat :=
  obj.factors.size

end ProductObject

/-- References a specific factor flowing into or out of a block. -/
structure BlockAttachment where
  block : Nat
  factor : Nat
  deriving Repr, DecidableEq, Inhabited, BEq

namespace BlockAttachment

def shift (blockOffset factorOffset : Nat) (a : BlockAttachment) : BlockAttachment :=
  { block := a.block + blockOffset, factor := a.factor + factorOffset }

end BlockAttachment

/-- Internal wiring between blocks inside a product morphism. -/
structure Wiring where
  source : BlockAttachment
  target : BlockAttachment
  deriving Repr, DecidableEq, Inhabited, BEq

namespace Wiring

def shiftBlocks (offset : Nat) (w : Wiring) : Wiring :=
  { source := w.source.shift offset 0, target := w.target.shift offset 0 }

end Wiring

/--
Product-category morphism skeleton with explicit block placements and wiring.
Later phases refine the semantics of the blocks and remappings.
-/
structure ProductMorphism where
  dom : ProductObject
  cod : ProductObject
  blocks : Array MorphismTerm := #[]
  blockInputs : Array BlockAttachment := #[]
  blockOutputs : Array BlockAttachment := #[]
  explicitWiring : Array Wiring := #[]
  deriving Repr

namespace ProductMorphism

def id (obj : ProductObject) : ProductMorphism :=
  { dom := obj, cod := obj }

def blockCount (hom : ProductMorphism) : Nat :=
  hom.blocks.size

def sequential (f g : ProductMorphism) : ProductMorphism :=
  { dom := f.dom
    cod := g.cod
    blocks := f.blocks ++ g.blocks
    blockInputs := f.blockInputs ++ g.blockInputs
    blockOutputs := f.blockOutputs ++ g.blockOutputs
    explicitWiring := f.explicitWiring ++ g.explicitWiring }

def parallel (f g : ProductMorphism) : ProductMorphism :=
  let blockOffset := f.blocks.size
  let domFactorOffset := f.dom.factorCount
  let codFactorOffset := f.cod.factorCount
  { dom := f.dom.append g.dom
    cod := f.cod.append g.cod
    blocks := f.blocks ++ g.blocks
    blockInputs := f.blockInputs ++ g.blockInputs.map (fun a => a.shift blockOffset domFactorOffset)
    blockOutputs := f.blockOutputs ++ g.blockOutputs.map (fun a => a.shift blockOffset codFactorOffset)
    explicitWiring := f.explicitWiring ++ g.explicitWiring.map (Wiring.shiftBlocks blockOffset) }

end ProductMorphism

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra
