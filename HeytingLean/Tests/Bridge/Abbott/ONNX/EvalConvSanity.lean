import HeytingLean.Bridge.Abbott.ONNX.Eval.Convolution

namespace HeytingLean.Tests.Bridge.Abbott.ONNX.EvalConvSanity

open HeytingLean.Bridge.Abbott.ONNX

def baseEnv : Env :=
  [ ("x", .ints [1, 1, 3, 3] [1, 2, 3, 4, 5, 6, 7, 8, 9])
  , ("kernel", .ints [1, 1, 2, 2] [1, 0, 0, 1])
  , ("bias", .ints [1] [1])
  , ("fx", .floats [1, 1, 2, 2] [1.0, 2.0, 3.0, 4.0])
  , ("scale", .floats [1] [1.0])
  , ("fbias", .floats [1] [0.0])
  , ("mean", .floats [1] [1.0])
  , ("var", .floats [1] [1.0])
  ]

def evalOutput (node : Node) (name : String := "out") : Except String TensorValue := do
  let env ← EvalConvolution.evalNode node baseEnv
  Env.lookup env name

example :
    (evalOutput
      { op := .conv
        inputs := ["x", "kernel", "bias"]
        outputs := ["out"] } >>= TensorValue.expectInts) =
      .ok [7, 9, 13, 15] := rfl

example :
    (match evalOutput
      { op := .batchNormalization
        inputs := ["fx", "scale", "fbias", "mean", "var"]
        outputs := ["out"]
        attrs := [("epsilon", .float 0.0)] } >>= TensorValue.expectFloats with
    | .ok values => values == [0.0, 1.0, 2.0, 3.0]
    | .error _ => false) = true := by native_decide

example :
    (evalOutput
      { op := .relu
        inputs := ["x"]
        outputs := ["out"] } >>= TensorValue.expectInts) =
      .ok [1, 2, 3, 4, 5, 6, 7, 8, 9] := rfl

example :
    (evalOutput
      { op := .clip
        inputs := ["x"]
        outputs := ["out"]
        attrs := [("min", .int 3), ("max", .int 7)] } >>= TensorValue.expectInts) =
      .ok [3, 3, 3, 4, 5, 6, 7, 7, 7] := rfl

example :
    (evalOutput
      { op := .reduceSum
        inputs := ["x"]
        outputs := ["out"]
        attrs := [("axes", .nats [2, 3]), ("keepdims", .int 0)] } >>= TensorValue.expectInts) =
      .ok [45] := rfl

example :
    (match evalOutput
      { op := .reduceMean
        inputs := ["fx"]
        outputs := ["out"]
        attrs := [("axes", .nats [2, 3]), ("keepdims", .int 0)] } >>= TensorValue.expectFloats with
    | .ok values => values == [2.5]
    | .error _ => false) = true := by native_decide

example :
    (evalOutput
      { op := .maxPool
        inputs := ["x"]
        outputs := ["out"]
        attrs := [("kernel_shape", .nats [2, 2]), ("strides", .nats [1, 1])] } >>= TensorValue.expectInts) =
      .ok [5, 6, 8, 9] := rfl

example :
    (match evalOutput
      { op := .averagePool
        inputs := ["fx"]
        outputs := ["out"]
        attrs := [("kernel_shape", .nats [2, 2]), ("strides", .nats [1, 1])] } >>= TensorValue.expectFloats with
    | .ok values => values == [2.5]
    | .error _ => false) = true := by native_decide

example :
    (match evalOutput
      { op := .pad
        inputs := ["x"]
        outputs := ["out"]
        attrs := [("pads", .nats [0, 0, 1, 1, 0, 0, 1, 1]), ("value", .int 0)] } >>= TensorValue.expectInts with
    | .ok values => values == [0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0]
    | .error _ => false) = true := by native_decide

end HeytingLean.Tests.Bridge.Abbott.ONNX.EvalConvSanity
