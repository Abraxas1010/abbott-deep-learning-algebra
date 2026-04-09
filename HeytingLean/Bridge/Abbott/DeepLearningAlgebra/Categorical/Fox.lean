import Mathlib.CategoryTheory.Types.Basic
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Laws

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.FoundationalLaws

abbrev FoxTupleObject (n : Nat) (α : Type) := TensorTuple n α

def foxFreeMorphism (components : Fin n → σ → α) : σ ⟶ FoxTupleObject n α :=
  ↾(foxFreeConstruction components)

def projectionMorphism (i : Fin n) : FoxTupleObject n α ⟶ α :=
  ↾(fun xs => xs i)

theorem fox_projection (components : Fin n → σ → α) (i : Fin n) :
    foxFreeMorphism components ≫ projectionMorphism i = components i := by
  funext x
  rfl

theorem fox_uniqueness {f g : σ ⟶ FoxTupleObject n α}
    (h : ∀ i : Fin n, f ≫ projectionMorphism i = g ≫ projectionMorphism i) :
    f = g := by
  funext x
  funext i
  exact congrFun (h i) x

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical
