import Mathlib.Data.Matrix.Basic
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Stride

namespace HeytingLean.Tests.Bridge.Abbott.Categorical.StrideSanity

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

def batch : Axis := { uid := 0, label := "batch", size := 4 }
def channel : Axis := { uid := 1, label := "channel", size := 8 }

def pairAxes : AxisProduct := { axes := #[batch, channel] }

def affineF : StrideHom pairAxes pairAxes where
  linear := fun i j =>
    match i.1, j.1 with
    | 0, 0 => 1
    | 0, 1 => 2
    | 1, 0 => 0
    | 1, 1 => 1
    | _, _ => 0
  offset := fun i =>
    match i.1 with
    | 0 => 3
    | 1 => -1
    | _ => 0

def affineG : StrideHom pairAxes pairAxes where
  linear := fun i j =>
    match i.1, j.1 with
    | 0, 0 => 2
    | 0, 1 => 1
    | 1, 0 => 1
    | 1, 1 => 0
    | _, _ => 0
  offset := fun i =>
    match i.1 with
    | 0 => 4
    | 1 => 5
    | _ => 0

example : (StrideHom.comp affineF affineG).linear ⟨0, by decide⟩ ⟨0, by decide⟩ = 4 := by
  native_decide

example : (StrideHom.comp affineF affineG).linear ⟨0, by decide⟩ ⟨1, by decide⟩ = 1 := by
  native_decide

example : (StrideHom.comp affineF affineG).linear ⟨1, by decide⟩ ⟨0, by decide⟩ = 1 := by
  native_decide

example : (StrideHom.comp affineF affineG).offset ⟨0, by decide⟩ = 17 := by
  native_decide

example : (StrideHom.comp affineF affineG).offset ⟨1, by decide⟩ = 4 := by
  native_decide

example : (StrideHom.comp affineF affineG).toMor.dom = pairAxes := rfl

example : (StrideHom.comp affineF affineG).toMor.cod = pairAxes := rfl

end HeytingLean.Tests.Bridge.Abbott.Categorical.StrideSanity
