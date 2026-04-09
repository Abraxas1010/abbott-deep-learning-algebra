import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Fox

namespace HeytingLean.Tests.Bridge.Abbott.Categorical.FoxSanity

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

def foxComponents : Fin 2 → Nat → Nat
  | ⟨0, _⟩, x => x
  | ⟨1, _⟩, x => x + 1

example :
    foxFreeMorphism foxComponents ≫ projectionMorphism ⟨1, by decide⟩ =
      foxComponents ⟨1, by decide⟩ := by
  exact fox_projection foxComponents ⟨1, by decide⟩

def foxFamilyA : Nat ⟶ FoxTupleObject 2 Nat :=
  foxFreeMorphism foxComponents

def foxFamilyB : Nat ⟶ FoxTupleObject 2 Nat :=
  ↾(fun x i => foxComponents i x)

example : foxFamilyA = foxFamilyB := by
  apply fox_uniqueness
  intro i
  funext x
  cases i using Fin.cases with
  | zero =>
      rfl
  | succ j =>
      cases j using Fin.cases with
      | zero =>
          rfl
      | succ j =>
          exact Fin.elim0 j

end HeytingLean.Tests.Bridge.Abbott.Categorical.FoxSanity
