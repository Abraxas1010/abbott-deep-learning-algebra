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

def segmentSizes2 : Fin 2 → Nat
  | ⟨0, _⟩ => 2
  | ⟨1, _⟩ => 1

def tuple3 : TensorTuple 3 Nat
  | ⟨0, _⟩ => 10
  | ⟨1, _⟩ => 20
  | ⟨2, _⟩ => 30

def expected3 : TensorTuple 3 Nat
  | ⟨0, _⟩ => 20
  | ⟨1, _⟩ => 10
  | ⟨2, _⟩ => 30

def directSumExample : FiniteRemapping 3 3 := by
  simpa [segmentSizes2, FiniteRemapping.segmentedArity] using
    (FiniteRemapping.directSumFamily segmentSizes2 segmentSizes2
      (fun
        | ⟨0, _⟩ => swapTwo
        | ⟨1, _⟩ => FiniteRemapping.id 1))

def segmentSizesSwap : Fin 2 → Nat
  | ⟨0, _⟩ => 2
  | ⟨1, _⟩ => 3

def tuple5 : TensorTuple 5 Nat
  | ⟨0, _⟩ => 10
  | ⟨1, _⟩ => 11
  | ⟨2, _⟩ => 20
  | ⟨3, _⟩ => 21
  | ⟨4, _⟩ => 22

def expected5 : TensorTuple 5 Nat
  | ⟨0, _⟩ => 20
  | ⟨1, _⟩ => 21
  | ⟨2, _⟩ => 22
  | ⟨3, _⟩ => 10
  | ⟨4, _⟩ => 11

def flatSwap : FiniteRemapping 5 5 := by
  simpa [segmentSizesSwap, FiniteRemapping.segmentedArity] using
    (FiniteRemapping.flatRemapping swapTwo segmentSizesSwap)

def swapStride : StrideMor :=
  StrideMor.purePermutation pairAxes pairAxes swapTwo

def swapReindexing : Reindexing :=
  { dom := pairObj, cod := pairObj, stride := swapStride }

def swapNatural : NaturalReindexing :=
  { reindexing := swapReindexing
    remap := swapTwo
    stride_eq := rfl }

def identityFamily : NaturalBroadcastFamily Nat Nat where
  lift := fun xs => xs
  natural := by
    intro dom cod μ xs
    rfl

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

example : remapTuple directSumExample tuple3 = expected3 := by
  funext i
  cases i using Fin.cases with
  | zero =>
      native_decide
  | succ i =>
      cases i using Fin.cases with
      | zero =>
          native_decide
      | succ i =>
          cases i using Fin.cases with
          | zero =>
              native_decide
          | succ i =>
              exact Fin.elim0 i

example : remapTuple flatSwap tuple5 = expected5 := by
  funext i
  cases i using Fin.cases with
  | zero =>
      native_decide
  | succ i =>
      cases i using Fin.cases with
      | zero =>
          native_decide
      | succ i =>
          cases i using Fin.cases with
          | zero =>
              native_decide
          | succ i =>
              cases i using Fin.cases with
              | zero =>
                  native_decide
              | succ i =>
                  cases i using Fin.cases with
                  | zero =>
                      native_decide
                  | succ i =>
                      exact Fin.elim0 i

example :
    swapNatural.asTupleMap tuple2 = remapTuple swapTwo tuple2 := rfl

example :
    swapNatural.asTupleMap (identityFamily.lift tuple2) =
      identityFamily.lift (swapNatural.asTupleMap tuple2) := by
  simpa using swapNatural.yonedaSliding identityFamily tuple2

example :
    swapNatural.asTupleMap ((deterministicLiftFamily (fun n => n + 1)).lift tuple2) =
      (deterministicLiftFamily (fun n => n + 1)).lift (swapNatural.asTupleMap tuple2) := by
  simpa using swapNatural.yonedaSliding_ofDeterministic (fun n => n + 1) tuple2

end HeytingLean.Tests.Bridge.Abbott
