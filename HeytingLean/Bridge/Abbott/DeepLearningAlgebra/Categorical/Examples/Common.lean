import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Broadcast

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples

open HeytingLean.Bridge.Abbott.DeepLearningAlgebra

def eraseAxisLabels (axis : Axis) : Axis :=
  { uid := axis.uid
    label := ""
    size := axis.size }

def eraseArrayLabels (obj : ArrayObject) : ArrayObject :=
  { dtype := obj.dtype
    axes := { axes := obj.axes.axes.map eraseAxisLabels } }

def templateProductHom (template : ProductTemplate) : ProductHom template.dom template.cod where
  toMorphism := template.asProductMorphism
  dom_eq := rfl
  cod_eq := rfl

def castProductHom {X X' Y Y' : ProductObject} (f : ProductHom X Y)
    (hX : X = X') (hY : Y = Y') : ProductHom X' Y' := by
  cases hX
  cases hY
  exact f

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples
