import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.ArrayBroadcasted
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Elemental

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra

/-- Reindexing packages an array morphism driven by an explicit stride map. -/
structure Reindexing where
  dom : ArrayObject
  cod : ArrayObject
  stride : StrideMor
  deriving Repr

/-- Add a fresh batch axis in front of an array object. -/
structure BatchLift where
  batchAxis : Axis
  lifted : ArrayObject
  base : ArrayObject
  deriving Repr

namespace BatchLift

def valid (lift : BatchLift) : Bool :=
  lift.lifted.rank = lift.base.rank + 1

end BatchLift

/-- A weave explicitly records how inputs are interleaved or un-interleaved. -/
structure Weave where
  source : AxisProduct
  target : AxisProduct
  slotOrder : Array Nat
  deriving Repr, Inhabited

namespace Weave

def identity (p : AxisProduct) : Weave :=
  { source := p, target := p, slotOrder := Array.ofFn (fun i : Fin p.rank => i.1) }

def valid (w : Weave) : Bool :=
  w.slotOrder.size = w.target.rank

end Weave

/-- Explicit broadcasted operation: no hidden backend broadcasting semantics. -/
structure BroadcastedOperation where
  name : String
  inputs : Array ArrayObject
  output : ArrayObject
  base : ProductTemplate
  reindexings : Array Reindexing := #[]
  batchLifts : Array BatchLift := #[]
  inputWeaves : Array Weave := #[]
  outputWeaves : Array Weave := #[]
  deriving Repr

namespace BroadcastedOperation

def totalAuxiliaryMorphisms (op : BroadcastedOperation) : Nat :=
  op.reindexings.size + op.batchLifts.size + op.inputWeaves.size + op.outputWeaves.size

def explicitBroadcasting (op : BroadcastedOperation) : Bool :=
  op.totalAuxiliaryMorphisms > 0

end BroadcastedOperation

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra
