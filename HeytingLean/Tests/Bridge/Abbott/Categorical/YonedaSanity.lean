import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Yoneda

namespace HeytingLean.Tests.Bridge.Abbott.Categorical.YonedaSanity

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical
open FoundationalLaws

def swapTwo : FiniteRemapping 2 2 :=
  { toFun := fun i =>
      if h : i.1 = 0 then
        ⟨1, by decide⟩
      else
        ⟨0, by decide⟩ }

def tuple2 : TensorTuple 2 Nat
  | ⟨0, _⟩ => 3
  | ⟨1, _⟩ => 5

example :
    (reindexHom (α := Nat) swapTwo ≫ deterministicHom 2 (fun n => n + 1)) tuple2 =
      (deterministicHom 2 (fun n => n + 1) ≫ reindexHom (α := Nat) swapTwo) tuple2 := by
  exact congrFun (yoneda_sliding swapTwo (fun n => n + 1)) tuple2

example :
    reindexHom (α := Nat) swapTwo ≫ deterministicHom 2 (fun n => n + 1) =
      deterministicHom 2 (fun n => n + 1) ≫ reindexHom (α := Nat) swapTwo := by
  exact yoneda_sliding swapTwo (fun n => n + 1)

end HeytingLean.Tests.Bridge.Abbott.Categorical.YonedaSanity
