import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Remapping

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra

/-- Named finite axis used by the paper's axis-stride category. -/
structure Axis where
  uid : Nat
  label : String
  size : Nat
  deriving Repr, DecidableEq, Inhabited, BEq

/-- Ordered axis product with finite shape metadata. -/
structure AxisProduct where
  axes : Array Axis
  deriving Repr, Inhabited

namespace AxisProduct

def rank (p : AxisProduct) : Nat :=
  p.axes.size

def shape (p : AxisProduct) : Array Nat :=
  p.axes.map Axis.size

def totalSize (p : AxisProduct) : Nat :=
  p.shape.foldl (· * ·) 1

end AxisProduct

/--
Finite affine stride morphism.
For a reindexing pullback `dom <- cod`, `linear[row][col]` gives the coefficient
of the `col`th target axis in the `row`th source axis, and `offset[row]` gives
the constant shift for the source axis. This matches Definition 8's affine
matrix together with Definition 10's contravariant reindexing action.
-/
structure StrideMor where
  dom : AxisProduct
  cod : AxisProduct
  linear : Array (Array Int)
  offset : Array Int
  deriving Repr

namespace StrideMor

def domRank (f : StrideMor) : Nat :=
  f.dom.rank

def codRank (f : StrideMor) : Nat :=
  f.cod.rank

def rowCount (f : StrideMor) : Nat :=
  f.linear.size

def wellShaped (f : StrideMor) : Bool :=
  f.rowCount = f.domRank &&
    f.linear.all (fun row => row.size = f.codRank) &&
    f.offset.size = f.domRank

def identity (p : AxisProduct) : StrideMor :=
  let mkRow := fun i =>
    Array.ofFn (fun j : Fin p.rank => if i = j.1 then (1 : Int) else 0)
  { dom := p
    cod := p
    linear := Array.ofFn (fun i : Fin p.rank => mkRow i.1)
    offset := Array.replicate p.rank 0 }

def purePermutation (dom cod : AxisProduct) (perm : FiniteRemapping dom.rank cod.rank) : StrideMor :=
  let mkRow := fun i =>
    Array.ofFn (fun j : Fin cod.rank =>
      if perm.toFun j = i then (1 : Int) else 0)
  { dom := dom
    cod := cod
    linear := Array.ofFn (fun i : Fin dom.rank => mkRow i)
    offset := Array.replicate dom.rank 0 }

end StrideMor

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra
