import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Array

namespace HeytingLean.Tests.Bridge.Abbott.Categorical.ArraySanity

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

def batch : Axis := { uid := 0, label := "batch", size := 4 }
def channel : Axis := { uid := 1, label := "channel", size := 8 }

def pairAxes : AxisProduct := { axes := #[batch, channel] }

def pairObj : ArrayObject := { dtype := .float, axes := pairAxes }

def strideF : StrideHom pairAxes pairAxes where
  linear := fun i j =>
    match i.1, j.1 with
    | 0, 0 => 1
    | 1, 1 => 1
    | _, _ => 0
  offset := fun _ => 0

def strideG : StrideHom pairAxes pairAxes where
  linear := fun i j =>
    match i.1, j.1 with
    | 0, 1 => 1
    | 1, 0 => 1
    | _, _ => 0
  offset := fun i =>
    match i.1 with
    | 0 => 2
    | 1 => 0
    | _ => 0

def payloadF : MorphismTerm := .primitive "relu" (ArrayObject.toObjectTerm pairObj) (ArrayObject.toObjectTerm pairObj)
def payloadG : MorphismTerm := .primitive "scale" (ArrayObject.toObjectTerm pairObj) (ArrayObject.toObjectTerm pairObj)

def homF : ArrayHom pairObj pairObj where
  indexMap := strideF
  payloadStages := [payloadF]

def homG : ArrayHom pairObj pairObj where
  indexMap := strideG
  payloadStages := [payloadG]

example : (ArrayHom.comp homF homG).toMor.shapeCompatible = true := by
  exact ArrayHom.shapeCompatible_comp homF homG

example :
    (ArrayHom.comp homF homG).toMor.payload = MorphismTerm.compose payloadG payloadF := by
  rfl

example :
    (ArrayHom.comp (ArrayHom.id pairObj) homF).payloadStages = homF.payloadStages := by
  rfl

end HeytingLean.Tests.Bridge.Abbott.Categorical.ArraySanity
