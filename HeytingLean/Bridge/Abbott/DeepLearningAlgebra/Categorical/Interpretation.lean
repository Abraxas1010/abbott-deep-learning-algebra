import Mathlib.CategoryTheory.PathCategory.Basic
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Syntax
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Product
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Broadcast

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra

def interpObject : ObjectTerm → ProductObject
  | .product factors => { factors := factors }
  | obj => ProductObject.singleton obj

structure PresentedProductObject where
  carrier : ProductObject
  deriving Repr

structure PresentedArrow (X Y : PresentedProductObject) where
  term : MorphismTerm
  interpreted : ProductHom X.carrier Y.carrier

def interpPresentedArrow (f : PresentedArrow X Y) : ProductHom X.carrier Y.carrier :=
  f.interpreted

instance : Quiver PresentedProductObject where
  Hom X Y := PresentedArrow X Y

abbrev PresentedProductCat := CategoryTheory.Paths PresentedProductObject

def presentedProductPrefunctor : PresentedProductObject ⥤q ProductObject where
  obj X := X.carrier
  map f := interpPresentedArrow f

def interpToProduct : PresentedProductCat ⥤ ProductObject :=
  CategoryTheory.Paths.lift presentedProductPrefunctor

theorem interp_map_id (X : PresentedProductCat) :
    interpToProduct.map (𝟙 X) = 𝟙 (interpToProduct.obj X) := by
  simp [interpToProduct]

theorem interpToProduct_toPath (f : PresentedArrow X Y) :
    interpToProduct.map (Quiver.Hom.toPath f) = f.interpreted := by
  exact
    CategoryTheory.Paths.lift_toPath presentedProductPrefunctor f

theorem interp_map_comp {X Y Z : PresentedProductCat} (f : X ⟶ Y) (g : Y ⟶ Z) :
    interpToProduct.map (f ≫ g) = interpToProduct.map f ≫ interpToProduct.map g := by
  simp [interpToProduct]

structure PresentedBroadcastObject where
  carrier : BroadcastObject
  deriving Repr

namespace PresentedBroadcastObject

def singleton (obj : ArrayObject) : PresentedBroadcastObject :=
  { carrier := BroadcastObject.singleton obj }

end PresentedBroadcastObject

inductive PresentedBroadcastArrow : PresentedBroadcastObject → PresentedBroadcastObject → Type where
  | base {X Y : PresentedBroadcastObject}
      (term : MorphismTerm)
      (f : ProductHom X.carrier.toProductObject Y.carrier.toProductObject) :
      PresentedBroadcastArrow X Y
  | reindexing (η : Reindexing) :
      PresentedBroadcastArrow
        (PresentedBroadcastObject.singleton η.dom)
        (PresentedBroadcastObject.singleton η.cod)
  | batchLift (β : BatchLift) (hValid : β.valid = true) :
      PresentedBroadcastArrow
        (PresentedBroadcastObject.singleton β.base)
        (PresentedBroadcastObject.singleton β.lifted)
  | weave (τ : BaseType) (w : Weave) (hValid : w.valid = true) :
      PresentedBroadcastArrow
        { carrier := BroadcastObject.singleton { dtype := τ, axes := w.source } }
        { carrier := BroadcastObject.singleton { dtype := τ, axes := w.target } }

def interpPresentedBroadcastArrow (f : PresentedBroadcastArrow X Y) :
    BroadcastHom X.carrier Y.carrier := by
  cases f with
  | base _ g => exact BroadcastHom.ofBase g
  | reindexing η => exact BroadcastHom.ofReindexing η
  | batchLift β hValid => exact BroadcastHom.ofBatchLift β hValid
  | weave τ w hValid => exact BroadcastHom.ofWeave τ w hValid

instance : Quiver PresentedBroadcastObject where
  Hom X Y := PresentedBroadcastArrow X Y

abbrev PresentedBroadcastCat := CategoryTheory.Paths PresentedBroadcastObject

def presentedBroadcastPrefunctor : PresentedBroadcastObject ⥤q BroadcastCat where
  obj X := X.carrier
  map f := interpPresentedBroadcastArrow f

def interpToBroadcast : PresentedBroadcastCat ⥤ BroadcastCat :=
  CategoryTheory.Paths.lift presentedBroadcastPrefunctor

theorem interpToBroadcast_toPath_reindexing (η : Reindexing) :
    interpToBroadcast.map (Quiver.Hom.toPath (PresentedBroadcastArrow.reindexing η)) =
      BroadcastHom.ofReindexing η := by
  rfl

theorem reindexing_reflects_stride (η : Reindexing) :
    interpToBroadcast.map (Quiver.Hom.toPath (PresentedBroadcastArrow.reindexing η)) =
      BroadcastHom.ofReindexing η :=
  interpToBroadcast_toPath_reindexing η

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical
