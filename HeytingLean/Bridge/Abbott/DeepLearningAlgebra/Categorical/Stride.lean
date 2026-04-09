import Mathlib.CategoryTheory.Category.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.AxisStride

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

open CategoryTheory
open Matrix
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra

structure StrideHom (X Y : AxisProduct) where
  linear : Matrix (Fin X.rank) (Fin Y.rank) Int
  offset : Fin X.rank → Int

namespace StrideHom

def toMor (f : StrideHom X Y) : StrideMor :=
  { dom := X
    cod := Y
    linear := Array.ofFn (fun i : Fin X.rank =>
      Array.ofFn (fun j : Fin Y.rank => f.linear i j))
    offset := Array.ofFn (fun i : Fin X.rank => f.offset i) }

@[simp] theorem toMor_dom (f : StrideHom X Y) : f.toMor.dom = X := rfl

@[simp] theorem toMor_cod (f : StrideHom X Y) : f.toMor.cod = Y := rfl

@[simp] def id (X : AxisProduct) : StrideHom X X where
  linear := 1
  offset := 0

@[simp] def comp (f : StrideHom X Y) (g : StrideHom Y Z) : StrideHom X Z where
  linear := f.linear * g.linear
  offset := f.offset + f.linear *ᵥ g.offset

@[ext] theorem ext (f g : StrideHom X Y)
    (hLinear : f.linear = g.linear)
    (hOffset : f.offset = g.offset) : f = g := by
  cases f
  cases g
  cases hLinear
  cases hOffset
  rfl

@[simp] theorem id_comp (f : StrideHom X Y) : comp (id X) f = f := by
  apply ext
  · simp [comp, id]
  · simp [comp, id]

@[simp] theorem comp_id (f : StrideHom X Y) : comp f (id Y) = f := by
  apply ext
  · simp [comp, id]
  · simp [comp, id]

@[simp] theorem assoc (f : StrideHom W X) (g : StrideHom X Y) (h : StrideHom Y Z) :
    comp (comp f g) h = comp f (comp g h) := by
  apply ext
  · simp [comp, Matrix.mul_assoc]
  · funext i
    simp [comp, Matrix.mulVec_add, Matrix.mulVec_mulVec, add_assoc]

instance : Category AxisProduct where
  Hom X Y := StrideHom X Y
  id := StrideHom.id
  comp f g := StrideHom.comp f g
  id_comp := by
    intro X Y f
    exact id_comp f
  comp_id := by
    intro X Y f
    exact comp_id f
  assoc := by
    intro W X Y Z f g h
    exact assoc f g h

end StrideHom

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical
