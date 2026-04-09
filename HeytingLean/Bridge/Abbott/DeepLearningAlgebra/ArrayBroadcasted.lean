import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.AxisStride

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra

/-- Array object in the broadcasted category: dtype plus explicit axes. -/
structure ArrayObject where
  dtype : BaseType
  axes : AxisProduct
  deriving Repr, Inhabited

namespace ArrayObject

def rank (obj : ArrayObject) : Nat :=
  obj.axes.rank

def shape (obj : ArrayObject) : Array Nat :=
  obj.axes.shape

end ArrayObject

/-- Separates semantic axes from storage positions. -/
structure ArraySeparator where
  semanticAxis : Nat
  storageAxis : Nat
  deriving Repr, DecidableEq, Inhabited, BEq

/-- Witness that a separator family covers a full object. -/
structure ArrayIndexWitness where
  object : ArrayObject
  separators : Array ArraySeparator
  deriving Repr, Inhabited

namespace ArrayIndexWitness

def complete (w : ArrayIndexWitness) : Bool :=
  w.separators.size = w.object.rank

end ArrayIndexWitness

/-- Morphisms in the array-broadcasted category. -/
structure ArrayMorphism where
  dom : ArrayObject
  cod : ArrayObject
  indexMap : StrideMor
  payload : MorphismTerm
  deriving Repr

namespace ArrayMorphism

def shapeCompatible (f : ArrayMorphism) : Bool :=
  f.indexMap.domRank = f.dom.rank && f.indexMap.codRank = f.cod.rank

end ArrayMorphism

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra
