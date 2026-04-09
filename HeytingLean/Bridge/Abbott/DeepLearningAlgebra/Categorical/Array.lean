import Mathlib.CategoryTheory.Category.Basic
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.ArrayBroadcasted
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Stride

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra

namespace Axis

def toSlotRef (axis : Axis) : SlotRef :=
  { uid := axis.uid
    kind := .axis
    label := axis.label }

end Axis

namespace ArrayObject

def toObjectTerm (obj : ArrayObject) : ObjectTerm :=
  .tensor obj.dtype (obj.axes.axes.map Axis.toSlotRef)

end ArrayObject

structure ArrayHom (X Y : ArrayObject) where
  indexMap : StrideHom X.axes Y.axes
  payloadStages : List MorphismTerm := []

namespace ArrayHom

def normalizePayload (dom : ArrayObject) : List MorphismTerm → MorphismTerm
  | [] => .id (ArrayObject.toObjectTerm dom)
  | t :: ts => ts.foldl (fun acc u => .compose u acc) t

def toMor (f : ArrayHom X Y) : ArrayMorphism :=
  { dom := X
    cod := Y
    indexMap := f.indexMap.toMor
    payload := normalizePayload X f.payloadStages }

@[simp] theorem toMor_dom (f : ArrayHom X Y) : f.toMor.dom = X := rfl

@[simp] theorem toMor_cod (f : ArrayHom X Y) : f.toMor.cod = Y := rfl

@[simp] theorem shapeCompatible (f : ArrayHom X Y) : f.toMor.shapeCompatible = true := by
  simp [toMor, ArrayMorphism.shapeCompatible, StrideMor.domRank, StrideMor.codRank,
    ArrayObject.rank, AxisProduct.rank]

@[simp] def id (X : ArrayObject) : ArrayHom X X where
  indexMap := 𝟙 X.axes
  payloadStages := []

@[simp] def comp (f : ArrayHom X Y) (g : ArrayHom Y Z) : ArrayHom X Z where
  indexMap := StrideHom.comp f.indexMap g.indexMap
  payloadStages := f.payloadStages ++ g.payloadStages

@[ext] theorem ext (f g : ArrayHom X Y)
    (hIndex : f.indexMap = g.indexMap)
    (hPayload : f.payloadStages = g.payloadStages) : f = g := by
  cases f
  cases g
  cases hIndex
  cases hPayload
  rfl

@[simp] theorem id_comp (f : ArrayHom X Y) : comp (id X) f = f := by
  apply ext
  · exact StrideHom.id_comp f.indexMap
  · simp [comp, id]

@[simp] theorem comp_id (f : ArrayHom X Y) : comp f (id Y) = f := by
  apply ext
  · exact StrideHom.comp_id f.indexMap
  · simp [comp, id]

@[simp] theorem assoc (f : ArrayHom W X) (g : ArrayHom X Y) (h : ArrayHom Y Z) :
    comp (comp f g) h = comp f (comp g h) := by
  apply ext
  · exact StrideHom.assoc f.indexMap g.indexMap h.indexMap
  · simp [comp, List.append_assoc]

instance : Category ArrayObject where
  Hom X Y := ArrayHom X Y
  id := ArrayHom.id
  comp f g := ArrayHom.comp f g
  id_comp := by
    intro X Y f
    exact id_comp f
  comp_id := by
    intro X Y f
    exact comp_id f
  assoc := by
    intro W X Y Z f g h
    exact assoc f g h

@[simp] theorem shapeCompatible_id (X : ArrayObject) :
    (id X).toMor.shapeCompatible = true :=
  shapeCompatible (id X)

@[simp] theorem shapeCompatible_comp (f : ArrayHom X Y) (g : ArrayHom Y Z) :
    (comp f g).toMor.shapeCompatible = true :=
  shapeCompatible (comp f g)

end ArrayHom

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical
