import HeytingLean.Bridge.Abbott.ONNX.Syntax

namespace HeytingLean.Bridge.Abbott.ONNX

namespace DType

def isFloat : DType → Bool
  | .float32 | .float64 => true
  | _ => false

def isIntegral : DType → Bool
  | .int64 => true
  | _ => false

def isBool : DType → Bool
  | .bool => true
  | _ => false

end DType

namespace TensorTy

def rank (ty : TensorTy) : Nat :=
  ty.shape.rank

def knownDims? (ty : TensorTy) : Option (List Nat) :=
  ty.shape.knownDims?

end TensorTy

namespace ValueDecl

def rank (decl : ValueDecl) : Nat :=
  decl.ty.rank

def knownDims? (decl : ValueDecl) : Option (List Nat) :=
  decl.ty.knownDims?

end ValueDecl

end HeytingLean.Bridge.Abbott.ONNX
