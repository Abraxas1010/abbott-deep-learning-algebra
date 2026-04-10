import HeytingLean.Bridge.Abbott.ONNX.Eval.Core

namespace HeytingLean.Tests.Bridge.Abbott.ONNX.EvalCoreSanity

open HeytingLean.Bridge.Abbott.ONNX

def baseEnv : Env :=
  [ ("x", .ints [2, 3] [1, 2, 3, 4, 5, 6])
  , ("bias", .ints [1, 3] [10, 20, 30])
  , ("scale", .ints [2, 1] [2, 3])
  , ("bools", .bools [2, 3] [true, false, true, false, true, false])
  , ("shape_2x2", .ints [2] [2, 2])
  , ("indices", .ints [2] [2, 0])
  , ("y", .ints [1, 3] [100, 200, 300])
  , ("z", .ints [2, 1] [7, 8])
  ]

def evalOutput (node : Node) (name : String := "out") : Except String TensorValue := do
  let env ← evalCoreNode node baseEnv
  Env.lookup env name

example :
    (evalOutput { op := .identity, inputs := ["x"], outputs := ["out"] } |>.map (·.shape)) =
      .ok [2, 3] := rfl

example :
    (evalOutput { op := .shape, inputs := ["x"], outputs := ["out"] } >>= TensorValue.expectInts) =
      .ok [2, 3] := rfl

example :
    (evalOutput
      { op := .reshape
        inputs := ["x"]
        outputs := ["out"]
        attrs := [("shape", .shape (.ofKnown [3, 2]))] } |>.map (·.shape)) = .ok [3, 2] := rfl

example :
    (evalOutput
      { op := .flatten
        inputs := ["x"]
        outputs := ["out"]
        attrs := [("axis", .nat 1)] } |>.map (·.shape)) = .ok [2, 3] := rfl

example :
    (evalOutput
      { op := .unsqueeze
        inputs := ["x"]
        outputs := ["out"]
        attrs := [("axes", .nats [1])] } |>.map (·.shape)) = .ok [2, 1, 3] := rfl

example :
    (match evalOutput
      { op := .squeeze
        inputs := ["shape_2x2"]
        outputs := ["out"]
        attrs := [("axes", .nats [0])] } with
    | .ok _ => false
    | .error _ => true) = true := rfl

example :
    (evalOutput
      { op := .transpose
        inputs := ["x"]
        outputs := ["out"]
        attrs := [("perm", .nats [1, 0])] } >>= TensorValue.expectInts) =
      .ok [1, 4, 2, 5, 3, 6] := rfl

example :
    (evalOutput
      { op := .expand
        inputs := ["bias"]
        outputs := ["out"]
        attrs := [("shape", .shape (.ofKnown [2, 3]))] } >>= TensorValue.expectInts) =
      .ok [10, 20, 30, 10, 20, 30] := rfl

example :
    (evalOutput { op := .add, inputs := ["x", "bias"], outputs := ["out"] } >>= TensorValue.expectInts) =
      .ok [11, 22, 33, 14, 25, 36] := rfl

example :
    (evalOutput { op := .mul, inputs := ["x", "scale"], outputs := ["out"] } >>= TensorValue.expectInts) =
      .ok [2, 4, 6, 12, 15, 18] := rfl

example :
    (evalOutput
      { op := .constantOfShape
        inputs := ["shape_2x2"]
        outputs := ["out"]
        attrs := [("value", .int 5)] } >>= TensorValue.expectInts) =
      .ok [5, 5, 5, 5] := rfl

example :
    (evalOutput
      { op := .cast
        inputs := ["bools"]
        outputs := ["out"]
        attrs := [("to", .dtype .int64)] } >>= TensorValue.expectInts) =
      .ok [1, 0, 1, 0, 1, 0] := rfl

example :
    (evalOutput
      { op := .where
        inputs := ["bools", "x", "y"]
        outputs := ["out"] } >>= TensorValue.expectInts) =
      .ok [1, 200, 3, 100, 5, 300] := rfl

example :
    (evalOutput
      { op := .concat
        inputs := ["x", "z"]
        outputs := ["out"]
        attrs := [("axis", .nat 1)] } >>= TensorValue.expectInts) =
      .ok [1, 2, 3, 7, 4, 5, 6, 8] := rfl

example :
    (evalOutput
      { op := .slice
        inputs := ["x"]
        outputs := ["out"]
        attrs := [("starts", .nats [0, 1]), ("ends", .nats [2, 3]), ("axes", .nats [0, 1])] }
      >>= TensorValue.expectInts) = .ok [2, 3, 5, 6] := rfl

example :
    (evalOutput
      { op := .gather
        inputs := ["x", "indices"]
        outputs := ["out"]
        attrs := [("axis", .nat 1)] } >>= TensorValue.expectInts) =
      .ok [3, 1, 6, 4] := rfl

end HeytingLean.Tests.Bridge.Abbott.ONNX.EvalCoreSanity
