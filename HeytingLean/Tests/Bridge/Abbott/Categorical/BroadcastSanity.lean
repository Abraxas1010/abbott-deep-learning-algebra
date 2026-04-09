import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Broadcast

namespace HeytingLean.Tests.Bridge.Abbott.Categorical.BroadcastSanity

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

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

def identityWeave : Weave :=
  Weave.identity pairAxes

def liftedAxes : AxisProduct := { axes := #[{ uid := 2, label := "outer", size := 2 }, batch, channel] }

def liftedObj : ArrayObject := { dtype := .float, axes := liftedAxes }

def batchLift : BatchLift :=
  { batchAxis := { uid := 2, label := "outer", size := 2 }
    lifted := liftedObj
    base := pairObj }

example :
    BroadcastHom.comp (BroadcastHom.ofReindexing swapReindexing)
      (BroadcastHom.id (BroadcastObject.singleton pairObj)) =
      BroadcastHom.ofReindexing swapReindexing := by
  exact BroadcastHom.ofReindexing_comp swapReindexing

example :
    BroadcastHom.comp
      (BroadcastHom.id (BroadcastObject.singleton pairObj))
      (BroadcastHom.ofReindexing swapReindexing) =
      BroadcastHom.ofReindexing swapReindexing := by
  exact BroadcastHom.id_comp (BroadcastHom.ofReindexing swapReindexing)

def mixedPath :
    BroadcastHom
      (BroadcastObject.singleton pairObj)
      (BroadcastObject.singleton liftedObj) :=
  BroadcastHom.comp
    (BroadcastHom.comp
      (BroadcastHom.ofReindexing swapReindexing)
      (BroadcastHom.ofWeave .float identityWeave (by native_decide)))
    (BroadcastHom.ofBatchLift batchLift (by native_decide))

example : mixedPath.length = 3 := by
  native_decide

example : batchLift.valid = true := by
  native_decide

end HeytingLean.Tests.Bridge.Abbott.Categorical.BroadcastSanity
