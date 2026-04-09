import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Laws

namespace HeytingLean.Tests.Bridge.Abbott

open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.FoundationalLaws

def foxComponents : Fin 2 → Nat → Nat
  | ⟨0, _⟩, x => x
  | ⟨1, _⟩, x => x + 1

def foxFamilyA : Nat → TensorTuple 2 Nat :=
  foxFreeConstruction foxComponents

def foxFamilyB : Nat → TensorTuple 2 Nat :=
  fun x i => foxComponents i x

example :
    remapTuple (projectionRemapping ⟨1, by decide⟩) (foxFreeConstruction foxComponents 7) =
      fun _ => 8 := by
  simpa using foxFreeConstruction_projection foxComponents 7 ⟨1, by decide⟩

example : foxFamilyA = foxFamilyB := by
  apply foxUniqueIdentification
  intro i x
  funext j
  simp [foxFamilyA, foxFamilyB, foxFreeConstruction, remapTuple, projectionRemapping]

end HeytingLean.Tests.Bridge.Abbott
