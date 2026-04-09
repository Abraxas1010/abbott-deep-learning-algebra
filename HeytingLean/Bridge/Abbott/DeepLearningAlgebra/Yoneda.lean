import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Laws

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra

namespace BroadcastYoneda

open FoundationalLaws

abbrev TensorTuple := FoundationalLaws.TensorTuple

def castTuple {n m : Nat} (h : n = m) (xs : TensorTuple n α) : TensorTuple m α :=
  fun i =>
    xs ⟨i.1, by
      exact h ▸ i.2⟩

structure DeterministicBroadcastMorphism (rank : Nat) (α : Type) (β : Type) where
  operation : BroadcastedOperation
  map : α → β

namespace DeterministicBroadcastMorphism

def eval (f : DeterministicBroadcastMorphism rank α β) (xs : TensorTuple rank α) :
    TensorTuple rank β :=
  pointwiseDeterministic f.map xs

end DeterministicBroadcastMorphism

def rankEndomorphism (η : NaturalReindexing)
    (rank_preserving : η.reindexing.cod.rank = η.reindexing.dom.rank)
    (xs : TensorTuple η.reindexing.dom.rank α) : TensorTuple η.reindexing.dom.rank α :=
  castTuple rank_preserving (η.asTupleMap xs)

theorem yonedaSliding_broadcast
    (η : NaturalReindexing)
    (rank_preserving : η.reindexing.cod.rank = η.reindexing.dom.rank)
    (f : DeterministicBroadcastMorphism η.reindexing.dom.rank α β)
    (xs : TensorTuple η.reindexing.dom.rank α) :
    rankEndomorphism η rank_preserving (f.eval xs) =
      f.eval (rankEndomorphism η rank_preserving xs) := by
  funext i
  simp [rankEndomorphism, castTuple, DeterministicBroadcastMorphism.eval,
    NaturalReindexing.asTupleMap, remapTuple, pointwiseDeterministic]

end BroadcastYoneda

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra
