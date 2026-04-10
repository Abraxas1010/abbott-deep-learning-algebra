import HeytingLean.Bridge.Abbott.ONNX.Eval.Attention

namespace HeytingLean.Tests.Bridge.Abbott.ONNX.EvalAttentionSanity

open HeytingLean.Bridge.Abbott.ONNX

def baseEnv : Env :=
  [ ("logits", .floats [2] [0.0, 0.0])
  , ("q", .floats [1, 2, 2] [0.0, 0.0, 0.0, 0.0])
  , ("k", .floats [1, 2, 2] [0.0, 0.0, 0.0, 0.0])
  , ("v", .floats [1, 2, 2] [1.0, 3.0, 5.0, 7.0])
  ]

def evalOutput (node : Node) (name : String := "out") : Except String TensorValue := do
  let env ← EvalAttention.evalNode node baseEnv
  Env.lookup env name

example :
    (match evalOutput
      { op := .softmax
        inputs := ["logits"]
        outputs := ["out"]
        attrs := [("axis", .nat 0)] } >>= TensorValue.expectFloats with
    | .ok values => values == [0.5, 0.5]
    | .error _ => false) = true := by native_decide

example :
    (evalOutput
      { op := .attention
        inputs := ["q", "k", "v"]
        outputs := ["out"]
        attrs := [("num_heads", .nat 1)] } |>.map (·.shape)) =
      .ok [1, 2, 2] := rfl

example :
    (match evalOutput
      { op := .attention
        inputs := ["q", "k", "v"]
        outputs := ["out"]
        attrs := [("num_heads", .nat 1)] } >>= TensorValue.expectFloats with
    | .ok values => values == [3.0, 5.0, 3.0, 5.0]
    | .error _ => false) = true := by native_decide

end HeytingLean.Tests.Bridge.Abbott.ONNX.EvalAttentionSanity
