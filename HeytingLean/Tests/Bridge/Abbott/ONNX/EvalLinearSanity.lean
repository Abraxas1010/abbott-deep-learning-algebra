import HeytingLean.Bridge.Abbott.ONNX.Eval.Linear

namespace HeytingLean.Tests.Bridge.Abbott.ONNX.EvalLinearSanity

open HeytingLean.Bridge.Abbott.ONNX

def baseEnv : Env :=
  [ ("a", .ints [2, 3] [1, 2, 3, 4, 5, 6])
  , ("b", .ints [3, 2] [7, 8, 9, 10, 11, 12])
  , ("bias", .ints [2] [1, 2])
  , ("fa", .floats [2, 2] [1.0, 2.0, 3.0, 4.0])
  , ("fb", .floats [2, 2] [0.5, 1.0, 1.5, 2.0])
  ]

def evalOutput (node : Node) (name : String := "out") : Except String TensorValue := do
  let env ← EvalLinear.evalNode node baseEnv
  Env.lookup env name

example :
    (evalOutput { op := .matMul, inputs := ["a", "b"], outputs := ["out"] } >>= TensorValue.expectInts) =
      .ok [58, 64, 139, 154] := rfl

example :
    (match evalOutput
      { op := .gemm
        inputs := ["a", "b", "bias"]
        outputs := ["out"] } >>= TensorValue.expectInts with
    | .ok values => values == [59, 66, 140, 156]
    | .error _ => false) = true := by native_decide

example :
    (match evalOutput
      { op := .gemm
        inputs := ["fa", "fb"]
        outputs := ["out"]
        attrs := [("transB", .int 1)] } >>= TensorValue.expectFloats with
    | .ok values => values == [2.5, 5.5, 5.5, 12.5]
    | .error _ => false) = true := by native_decide

end HeytingLean.Tests.Bridge.Abbott.ONNX.EvalLinearSanity
