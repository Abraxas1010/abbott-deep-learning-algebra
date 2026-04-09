import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Yoneda

namespace HeytingLean.Tests.Bridge.Abbott

open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.BroadcastYoneda

namespace YonedaSanity

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

def swapNatural : FoundationalLaws.NaturalReindexing :=
  { reindexing := swapReindexing
    remap := swapTwo
    stride_eq := rfl }

def tuple2 : BroadcastYoneda.TensorTuple 2 Nat
  | ⟨0, _⟩ => 3
  | ⟨1, _⟩ => 5

def pointwiseTemplate : ProductTemplate :=
  { name := "pointwise_increment"
    dom := ProductObject.singleton (.tensor .float #[])
    cod := ProductObject.singleton (.tensor .float #[])
    core := { hom := .primitive "pointwise_increment" (.tensor .float #[]) (.tensor .float #[]) } }

def pointwiseOp : BroadcastedOperation :=
  { name := "pointwise_increment"
    inputs := #[pairObj]
    output := pairObj
    base := pointwiseTemplate
    reindexings := #[swapReindexing] }

def plusOne : DeterministicBroadcastMorphism pairAxes.rank Nat Nat :=
  { operation := pointwiseOp
    map := fun n => n + 1 }

example : swapNatural.reindexing.cod.rank = swapNatural.reindexing.dom.rank := rfl

example :
    rankEndomorphism swapNatural rfl (plusOne.eval tuple2) =
      plusOne.eval (rankEndomorphism swapNatural rfl tuple2) := by
  simpa using yonedaSliding_broadcast swapNatural rfl plusOne tuple2

end YonedaSanity

end HeytingLean.Tests.Bridge.Abbott
