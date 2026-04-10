import HeytingLean.Bridge.Abbott.ONNX.Shape

namespace HeytingLean.Bridge.Abbott.ONNX

inductive TensorData where
  | floats (values : List Float)
  | ints (values : List Int)
  | bools (values : List Bool)
  deriving Repr, Inhabited, BEq

structure TensorValue where
  dtype : DType
  shape : List Nat
  data : TensorData
  deriving Repr, Inhabited, BEq

namespace TensorData

def length : TensorData → Nat
  | .floats values => values.length
  | .ints values => values.length
  | .bools values => values.length

end TensorData

namespace TensorValue

def elementCount (tensor : TensorValue) : Nat :=
  HeytingLean.Bridge.Abbott.ONNX.elementCount tensor.shape

def wellFormed (tensor : TensorValue) : Bool :=
  tensor.elementCount = tensor.data.length

def ints (shape : List Nat) (values : List Int) : TensorValue :=
  { dtype := .int64, shape := shape, data := .ints values }

def floats (shape : List Nat) (values : List Float) (dtype : DType := .float32) : TensorValue :=
  { dtype := dtype, shape := shape, data := .floats values }

def bools (shape : List Nat) (values : List Bool) : TensorValue :=
  { dtype := .bool, shape := shape, data := .bools values }

def expectInts (tensor : TensorValue) : Except String (List Int) :=
  match tensor.data with
  | .ints values => pure values
  | _ => .error s!"expected int tensor, got {reprStr tensor.dtype}"

def expectFloats (tensor : TensorValue) : Except String (List Float) :=
  match tensor.data with
  | .floats values => pure values
  | _ => .error s!"expected float tensor, got {reprStr tensor.dtype}"

def expectBools (tensor : TensorValue) : Except String (List Bool) :=
  match tensor.data with
  | .bools values => pure values
  | _ => .error s!"expected bool tensor, got {reprStr tensor.dtype}"

def reshape (tensor : TensorValue) (shape : List Nat) : Except String TensorValue := do
  let _ ← reshapeShape tensor.shape shape
  pure { tensor with shape := shape }

def withShape (tensor : TensorValue) (shape : List Nat) : Except String TensorValue := do
  if tensor.data.length = HeytingLean.Bridge.Abbott.ONNX.elementCount shape then
    pure { tensor with shape := shape }
  else
    .error "tensor payload length does not match requested shape"

def shapeTensor (tensor : TensorValue) : TensorValue :=
  .ints [tensor.shape.length] (tensor.shape.map Int.ofNat)

def constantOfShapeInts (shape : List Nat) (value : Int := 0) : TensorValue :=
  .ints shape (List.replicate (HeytingLean.Bridge.Abbott.ONNX.elementCount shape) value)

def constantOfShapeBools (shape : List Nat) (value : Bool := false) : TensorValue :=
  .bools shape (List.replicate (HeytingLean.Bridge.Abbott.ONNX.elementCount shape) value)

def cast (target : DType) (tensor : TensorValue) : Except String TensorValue :=
  match target, tensor.data with
  | .int64, .ints values =>
      pure { tensor with dtype := .int64, data := .ints values }
  | .int64, .bools values =>
      pure (.ints tensor.shape (values.map fun b => if b then 1 else 0))
  | .bool, .bools values =>
      pure { tensor with dtype := .bool, data := .bools values }
  | .bool, .ints values =>
      pure (.bools tensor.shape (values.map fun n => n ≠ 0))
  | .float32, .floats values =>
      pure { tensor with dtype := .float32, data := .floats values }
  | .float64, .floats values =>
      pure { tensor with dtype := .float64, data := .floats values }
  | .float32, .ints values =>
      pure (.floats tensor.shape (values.map fun n => Float.ofInt n) .float32)
  | .float64, .ints values =>
      pure (.floats tensor.shape (values.map fun n => Float.ofInt n) .float64)
  | _, _ =>
      .error s!"unsupported cast route: {reprStr tensor.dtype} -> {reprStr target}"

end TensorValue

end HeytingLean.Bridge.Abbott.ONNX
