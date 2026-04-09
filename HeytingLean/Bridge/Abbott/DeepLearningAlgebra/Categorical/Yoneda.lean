import Mathlib.CategoryTheory.Types.Basic
import Mathlib.CategoryTheory.Yoneda
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Interpretation
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Yoneda

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.FoundationalLaws

abbrev YonedaTupleObject (n : Nat) (α : Type) := TensorTuple n α

def reindexHom (μ : FiniteRemapping dom cod) : YonedaTupleObject dom α ⟶ YonedaTupleObject cod α :=
  ↾(remapTuple μ)

def deterministicHom (n : Nat) (f : α → β) : YonedaTupleObject n α ⟶ YonedaTupleObject n β :=
  ↾(pointwiseDeterministic f)

theorem yoneda_sliding (μ : FiniteRemapping dom cod) (f : α → β) :
    reindexHom (α := α) μ ≫ deterministicHom cod f =
      deterministicHom dom f ≫ reindexHom (α := β) μ := by
  funext xs
  exact deterministicNaturality μ f xs

theorem yoneda_reflection_helper (η : NaturalReindexing) (f : α → β) :
    reindexHom (α := α) η.remap ≫ deterministicHom η.reindexing.cod.rank f =
      deterministicHom η.reindexing.dom.rank f ≫ reindexHom (α := β) η.remap := by
  exact yoneda_sliding η.remap f

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical
