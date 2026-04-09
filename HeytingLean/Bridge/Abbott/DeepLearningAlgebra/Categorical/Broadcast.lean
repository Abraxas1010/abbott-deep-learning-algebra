import Mathlib.CategoryTheory.PathCategory.Basic
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Broadcasting
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Elemental
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Array
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Product

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra

structure BroadcastObject where
  factors : Array ArrayObject
  deriving Repr, Inhabited

namespace BroadcastObject

def singleton (obj : ArrayObject) : BroadcastObject :=
  { factors := #[obj] }

def factorCount (obj : BroadcastObject) : Nat :=
  obj.factors.size

def toProductObject (obj : BroadcastObject) : ProductObject :=
  { factors := obj.factors.map ArrayObject.toObjectTerm }

def tensorObj (X Y : BroadcastObject) : BroadcastObject :=
  { factors := X.factors ++ Y.factors }

end BroadcastObject

inductive BroadcastPrimitive : BroadcastObject → BroadcastObject → Type where
  | base {X Y : BroadcastObject} (f : ProductHom X.toProductObject Y.toProductObject) :
      BroadcastPrimitive X Y
  | reindexing (η : Reindexing) :
      BroadcastPrimitive (BroadcastObject.singleton η.dom) (BroadcastObject.singleton η.cod)
  | batchLift (β : BatchLift) (hValid : β.valid = true) :
      BroadcastPrimitive (BroadcastObject.singleton β.base) (BroadcastObject.singleton β.lifted)
  | weave (τ : BaseType) (w : Weave) (hValid : w.valid = true) :
      BroadcastPrimitive
        (BroadcastObject.singleton { dtype := τ, axes := w.source })
        (BroadcastObject.singleton { dtype := τ, axes := w.target })

instance : Quiver BroadcastObject where
  Hom X Y := BroadcastPrimitive X Y

abbrev BroadcastCat := CategoryTheory.Paths BroadcastObject

instance : Category BroadcastCat := CategoryTheory.Paths.categoryPaths BroadcastObject

abbrev BroadcastHom (X Y : BroadcastCat) := X ⟶ Y

namespace BroadcastHom

def id (X : BroadcastCat) : BroadcastHom X X :=
  𝟙 X

def comp (f : BroadcastHom X Y) (g : BroadcastHom Y Z) : BroadcastHom X Z :=
  f ≫ g

def ofBase (f : ProductHom X.toProductObject Y.toProductObject) :
    BroadcastHom X Y :=
  Quiver.Hom.toPath (BroadcastPrimitive.base f)

def ofReindexing (η : Reindexing) :
    BroadcastHom (BroadcastObject.singleton η.dom) (BroadcastObject.singleton η.cod) :=
  Quiver.Hom.toPath (BroadcastPrimitive.reindexing η)

def ofBatchLift (β : BatchLift) (hValid : β.valid = true) :
    BroadcastHom (BroadcastObject.singleton β.base) (BroadcastObject.singleton β.lifted) :=
  Quiver.Hom.toPath (BroadcastPrimitive.batchLift β hValid)

def ofWeave (τ : BaseType) (w : Weave) (hValid : w.valid = true) :
    BroadcastHom
      (BroadcastObject.singleton { dtype := τ, axes := w.source })
      (BroadcastObject.singleton { dtype := τ, axes := w.target }) :=
  Quiver.Hom.toPath (BroadcastPrimitive.weave τ w hValid)

@[simp] theorem id_comp (f : BroadcastHom X Y) : comp (id X) f = f := by
  simp [comp, id]

@[simp] theorem comp_id (f : BroadcastHom X Y) : comp f (id Y) = f := by
  simp [comp, id]

@[simp] theorem ofReindexing_comp (η : Reindexing) :
    comp (ofReindexing η) (id (BroadcastObject.singleton η.cod)) = ofReindexing η := by
  simp [comp, id, ofReindexing]

@[simp] theorem ofWeave_id (τ : BaseType) (w : Weave) (hValid : w.valid = true) :
    comp (ofWeave τ w hValid) (id (BroadcastObject.singleton { dtype := τ, axes := w.target })) =
      ofWeave τ w hValid := by
  simp [comp, id, ofWeave]

theorem batchLift_valid_respected (β : BatchLift) (hValid : β.valid = true) :
    β.valid = true := hValid

end BroadcastHom

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical
