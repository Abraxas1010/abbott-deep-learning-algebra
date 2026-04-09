import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.ProductCategory

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra

/-- Finite remappings model discrete index maps `Fin cod -> Fin dom`. -/
structure FiniteRemapping (dom cod : Nat) where
  toFun : Fin cod → Fin dom

namespace FiniteRemapping

def id (n : Nat) : FiniteRemapping n n :=
  { toFun := fun i => i }

def comp {a b c : Nat} (f : FiniteRemapping a b) (g : FiniteRemapping b c) :
    FiniteRemapping a c :=
  { toFun := fun i => f.toFun (g.toFun i) }

theorem comp_id {a b : Nat} (f : FiniteRemapping a b) :
    comp f (id b) = f := by
  cases f
  rfl

theorem id_comp {a b : Nat} (f : FiniteRemapping a b) :
    comp (id a) f = f := by
  cases f
  rfl

end FiniteRemapping

/-- Flatten a family of segment sizes into a single arity count. -/
def flattenedArity (segments : List Nat) : Nat :=
  segments.foldl (· + ·) 0

/-- Product block together with input/output remappings. -/
structure RemappedBlock where
  inDom : Nat
  inCod : Nat
  outDom : Nat
  outCod : Nat
  inputRemap : FiniteRemapping inDom inCod
  outputRemap : FiniteRemapping outDom outCod
  base : ProductMorphism

namespace RemappedBlock

def totalBoundarySlots (block : RemappedBlock) : Nat :=
  flattenedArity [block.inDom, block.inCod, block.outDom, block.outCod]

end RemappedBlock

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra
