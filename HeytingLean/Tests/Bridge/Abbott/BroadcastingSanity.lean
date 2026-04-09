import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.AxisStride
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.ArrayBroadcasted
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Broadcasting
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Laws

namespace HeytingLean.Tests.Bridge.Abbott

open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.FoundationalLaws

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

def tuple2 : TensorTuple 2 Nat
  | ⟨0, _⟩ => 3
  | ⟨1, _⟩ => 5

#check Axis
#check AxisProduct
#check StrideMor
#check ArrayObject
#check Reindexing
#check BatchLift
#check Weave
#check BroadcastedOperation

example : pairAxes.rank = 2 := rfl
example : pairAxes.shape = #[4, 8] := rfl
example : FoundationalLaws.strideIdentityWellShaped pairAxes = true := by
  native_decide
example : FoundationalLaws.weaveIdentityValid pairAxes = true := by
  native_decide

example :
    remapTuple swapTwo (pointwiseDeterministic (fun n => n + 1) tuple2) =
      pointwiseDeterministic (fun n => n + 1) (remapTuple swapTwo tuple2) := by
  simpa using deterministicNaturality swapTwo (fun n => n + 1) tuple2

example :
    remapTuple swapTwo ((deterministicLiftFamily (fun n => n + 1)).lift tuple2) =
      (deterministicLiftFamily (fun n => n + 1)).lift (remapTuple swapTwo tuple2) := by
  simpa using yonedaSliding_ofDeterministic swapTwo (fun n => n + 1) tuple2

end HeytingLean.Tests.Bridge.Abbott
