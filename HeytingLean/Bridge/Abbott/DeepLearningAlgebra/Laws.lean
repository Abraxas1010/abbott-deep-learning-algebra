import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Broadcasting

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra

namespace FoundationalLaws

def strideIdentityWellShaped (p : AxisProduct) : Bool :=
  (StrideMor.identity p).wellShaped

def weaveIdentityValid (p : AxisProduct) : Bool :=
  (Weave.identity p).valid

abbrev TensorTuple (n : Nat) (α : Type) := Fin n → α

def remapTuple {dom cod : Nat} (μ : FiniteRemapping dom cod) (xs : TensorTuple dom α) :
    TensorTuple cod α :=
  fun j => xs (μ.toFun j)

def pointwiseDeterministic (f : α → β) (xs : TensorTuple n α) : TensorTuple n β :=
  fun i => f (xs i)

theorem deterministicNaturality {dom cod : Nat} (μ : FiniteRemapping dom cod) (f : α → β)
    (xs : TensorTuple dom α) :
    remapTuple μ (pointwiseDeterministic f xs) =
      pointwiseDeterministic f (remapTuple μ xs) := by
  funext j
  rfl

def projectionRemapping {n : Nat} (i : Fin n) : FiniteRemapping n 1 :=
  { toFun := fun _ => i }

def foxFreeConstruction (components : Fin n → σ → α) : σ → TensorTuple n α :=
  fun x i => components i x

theorem foxFreeConstruction_projection {n : Nat} (components : Fin n → σ → α) (x : σ)
    (i : Fin n) :
    remapTuple (projectionRemapping i) (foxFreeConstruction components x) =
      fun _ => components i x := by
  funext j
  simp [remapTuple, projectionRemapping, foxFreeConstruction]

theorem foxUniqueIdentification {n : Nat} {f g : σ → TensorTuple n α}
    (h : ∀ (i : Fin n) (x : σ),
      remapTuple (projectionRemapping i) (f x) =
        remapTuple (projectionRemapping i) (g x)) :
    f = g := by
  funext x
  funext i
  have hi := congrFun (h i x) 0
  simpa [remapTuple, projectionRemapping] using hi

structure NaturalBroadcastFamily (α : Type) (β : Type) where
  lift : ∀ {n : Nat}, TensorTuple n α → TensorTuple n β
  natural : ∀ {dom cod : Nat} (μ : FiniteRemapping dom cod) (xs : TensorTuple dom α),
    remapTuple μ (lift xs) = lift (remapTuple μ xs)

def deterministicLiftFamily (f : α → β) : NaturalBroadcastFamily α β where
  lift := fun xs => pointwiseDeterministic f xs
  natural := by
    intro dom cod μ xs
    exact deterministicNaturality μ f xs

theorem yonedaSliding_ofNatural (family : NaturalBroadcastFamily α β)
    {dom cod : Nat} (μ : FiniteRemapping dom cod) (xs : TensorTuple dom α) :
    remapTuple μ (family.lift xs) = family.lift (remapTuple μ xs) :=
  family.natural μ xs

theorem yonedaSliding_ofDeterministic {dom cod : Nat} (μ : FiniteRemapping dom cod)
    (f : α → β) (xs : TensorTuple dom α) :
    remapTuple μ ((deterministicLiftFamily f).lift xs) =
      (deterministicLiftFamily f).lift (remapTuple μ xs) :=
  (deterministicLiftFamily f).natural μ xs

end FoundationalLaws

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra
