import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Interpretation

namespace HeytingLean.Tests.Bridge.Abbott.Categorical.InterpretationSanity

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

def objA : ObjectTerm := .atom { name := "A" }
def objB : ObjectTerm := .atom { name := "B" }
def objC : ObjectTerm := .atom { name := "C" }

def prodA : ProductObject := .singleton objA
def prodB : ProductObject := .singleton objB
def prodC : ProductObject := .singleton objC

def presA : PresentedProductObject := { carrier := prodA }
def presB : PresentedProductObject := { carrier := prodB }
def presC : PresentedProductObject := { carrier := prodC }

def edgeF : PresentedArrow presA presB where
  term := .primitive "f" objA objB
  interpreted :=
    { toMorphism :=
        { dom := prodA
          cod := prodB
          blocks := #[.primitive "f" objA objB]
          blockInputs := #[]
          blockOutputs := #[]
          explicitWiring := #[] }
      dom_eq := rfl
      cod_eq := rfl }

def edgeG : PresentedArrow presB presC where
  term := .primitive "g" objB objC
  interpreted :=
    { toMorphism :=
        { dom := prodB
          cod := prodC
          blocks := #[.primitive "g" objB objC]
          blockInputs := #[]
          blockOutputs := #[]
          explicitWiring := #[] }
      dom_eq := rfl
      cod_eq := rfl }

example :
    interpToProduct.map (Quiver.Hom.toPath edgeF ≫ Quiver.Hom.toPath edgeG) =
      interpToProduct.map (Quiver.Hom.toPath edgeF) ≫ interpToProduct.map (Quiver.Hom.toPath edgeG) := by
  exact interp_map_comp (Quiver.Hom.toPath edgeF) (Quiver.Hom.toPath edgeG)

example :
    interpToProduct.map (Quiver.Hom.toPath edgeF ≫ Quiver.Hom.toPath edgeG) =
      (edgeF.interpreted ≫ edgeG.interpreted) := by
  rw [interp_map_comp (Quiver.Hom.toPath edgeF) (Quiver.Hom.toPath edgeG)]
  simp [interpToProduct_toPath]

def batch : Axis := { uid := 0, label := "batch", size := 4 }
def channel : Axis := { uid := 1, label := "channel", size := 8 }
def pairAxes : AxisProduct := { axes := #[batch, channel] }
def pairObj : ArrayObject := { dtype := .float, axes := pairAxes }

def swapTwo : FiniteRemapping 2 2 :=
  { toFun := fun i =>
      if h : i.1 = 0 then
        ⟨1, by decide⟩
      else
        ⟨0, by decide⟩ }

def swapStride : StrideMor :=
  StrideMor.purePermutation pairAxes pairAxes swapTwo

def swapReindexing : Reindexing :=
  { dom := pairObj, cod := pairObj, stride := swapStride }

example :
    interpToBroadcast.map
        (Quiver.Hom.toPath (PresentedBroadcastArrow.reindexing swapReindexing)) =
      BroadcastHom.ofReindexing swapReindexing := by
  exact reindexing_reflects_stride swapReindexing

end HeytingLean.Tests.Bridge.Abbott.Categorical.InterpretationSanity
