import Lean

namespace HeytingLean.Bridge.Abbott.ONNX

open Lean

/-- Ranked tranche-1 ONNX operator surface owned by the Abbott fragment. -/
inductive RankedOp where
  | reshape
  | transpose
  | expand
  | unsqueeze
  | squeeze
  | add
  | mul
  | matMul
  | gemm
  | conv
  | attention
  | flatten
  | concat
  | slice
  | gather
  | shape
  | constantOfShape
  | batchNormalization
  | relu
  | clip
  | softmax
  | reduceSum
  | reduceMean
  | maxPool
  | averagePool
  | where
  | identity
  | cast
  | pad
  | resize
  deriving Repr, Inhabited, DecidableEq, BEq, ToJson, FromJson

def RankedOp.all : List RankedOp :=
  [ .reshape
  , .transpose
  , .expand
  , .unsqueeze
  , .squeeze
  , .add
  , .mul
  , .matMul
  , .gemm
  , .conv
  , .attention
  , .flatten
  , .concat
  , .slice
  , .gather
  , .shape
  , .constantOfShape
  , .batchNormalization
  , .relu
  , .clip
  , .softmax
  , .reduceSum
  , .reduceMean
  , .maxPool
  , .averagePool
  , .where
  , .identity
  , .cast
  , .pad
  , .resize
  ]

/-- Reduced dtype surface for the tranche-1 canonical fragment. -/
inductive DType where
  | float32
  | float64
  | int64
  | bool
  deriving Repr, Inhabited, DecidableEq, BEq, ToJson, FromJson

/-- Shape dimensions may be statically known or symbolic. -/
inductive ShapeDim where
  | known (size : Nat)
  | symbolic (name : String)
  deriving Repr, Inhabited, DecidableEq, BEq, ToJson, FromJson

structure ShapeExpr where
  dims : List ShapeDim
  deriving Repr, Inhabited, DecidableEq, BEq, ToJson, FromJson

structure TensorTy where
  dtype : DType
  shape : ShapeExpr
  deriving Repr, Inhabited, DecidableEq, BEq, ToJson, FromJson

inductive AttrValue where
  | int (value : Int)
  | ints (values : List Int)
  | nat (value : Nat)
  | nats (values : List Nat)
  | float (value : Float)
  | str (value : String)
  | bool (value : Bool)
  | dtype (value : DType)
  | shape (value : ShapeExpr)
  deriving Repr, Inhabited, ToJson, FromJson

abbrev AttrMap := List (String × AttrValue)

structure ValueDecl where
  name : String
  ty : TensorTy
  deriving Repr, Inhabited, ToJson, FromJson

structure Node where
  op : RankedOp
  inputs : List String
  outputs : List String
  attrs : AttrMap := []
  deriving Repr, Inhabited, ToJson, FromJson

structure FragmentGraph where
  inputs : List ValueDecl
  initializers : List ValueDecl := []
  nodes : List Node
  outputs : List ValueDecl
  deriving Repr, Inhabited, ToJson, FromJson

namespace ShapeExpr

def rank (shape : ShapeExpr) : Nat :=
  shape.dims.length

def ofKnown (dims : List Nat) : ShapeExpr :=
  { dims := dims.map ShapeDim.known }

def knownDims? (shape : ShapeExpr) : Option (List Nat) :=
  shape.dims.foldr
    (fun dim acc =>
      match dim, acc with
      | .known n, some ns => some (n :: ns)
      | _, _ => none)
    (some [])

end ShapeExpr

namespace AttrMap

def find? (attrs : AttrMap) (key : String) : Option AttrValue :=
  attrs.findSome? fun entry =>
    if entry.fst = key then some entry.snd else none

end AttrMap

theorem rankedOp_all_length : RankedOp.all.length = 30 := by
  native_decide

theorem rankedOp_all_nodup : RankedOp.all.Nodup := by
  native_decide

end HeytingLean.Bridge.Abbott.ONNX
