import Mathlib.CategoryTheory.Monoidal.Cartesian.Basic
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Broadcast

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

open HeytingLean.Bridge.Abbott.DeepLearningAlgebra

def broadcastTensorObj (X Y : BroadcastObject) : BroadcastObject :=
  BroadcastObject.tensorObj X Y

def tensorProductHom
    (f : ProductHom X.toProductObject Y.toProductObject)
    (g : ProductHom Z.toProductObject W.toProductObject) :
    ProductHom (broadcastTensorObj X Z).toProductObject (broadcastTensorObj Y W).toProductObject where
  toMorphism := ProductMorphism.parallel f.toMorphism g.toMorphism
  dom_eq := by
    simp [broadcastTensorObj, BroadcastObject.tensorObj, BroadcastObject.toProductObject,
      ProductMorphism.parallel, ProductObject.append, Array.map_append]
  cod_eq := by
    simp [broadcastTensorObj, BroadcastObject.tensorObj, BroadcastObject.toProductObject,
      ProductMorphism.parallel, ProductObject.append, Array.map_append]

def broadcastTensorHomBase
    (f : ProductHom X.toProductObject Y.toProductObject)
    (g : ProductHom Z.toProductObject W.toProductObject) :
    BroadcastHom (broadcastTensorObj X Z) (broadcastTensorObj Y W) :=
  BroadcastHom.ofBase (tensorProductHom f g)

theorem factorCount_broadcastTensorObj (X Y : BroadcastObject) :
    (broadcastTensorObj X Y).factorCount = X.factorCount + Y.factorCount := by
  simp [broadcastTensorObj, BroadcastObject.tensorObj, BroadcastObject.factorCount]

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical
