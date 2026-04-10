import HeytingLean.Bridge.Abbott.ONNX.Eval.Core

namespace HeytingLean.Bridge.Abbott.ONNX.EvalLinear

open HeytingLean.Bridge.Abbott.ONNX

private def requireArity (node : Node) (inputs outputs : Nat) : Except String Unit := do
  if node.inputs.length ≠ inputs then
    .error s!"{reprStr node.op} expected {inputs} inputs, got {node.inputs.length}"
  else if node.outputs.length ≠ outputs then
    .error s!"{reprStr node.op} expected {outputs} outputs, got {node.outputs.length}"
  else
    pure ()

private def getAt? (xs : List α) (idx : Nat) : Except String α :=
  match xs[idx]? with
  | some value => pure value
  | none => .error s!"index {idx} out of bounds"

private def transpose2DShape (shape : List Nat) : Except String (Nat × Nat) :=
  match shape with
  | [rows, cols] => pure (rows, cols)
  | _ => .error s!"expected rank-2 matrix, got shape {shape}"

private def transposeInts (shape : List Nat) (values : List Int) : Except String TensorValue := do
  let (rows, cols) ← transpose2DShape shape
  let outVals ← (List.range (rows * cols)).mapM fun flat => do
    let i := flat / rows
    let j := flat % rows
    getAt? values (j * cols + i)
  pure (.ints [cols, rows] outVals)

private def transposeFloats (shape : List Nat) (values : List Float) (dtype : DType) :
    Except String TensorValue := do
  let (rows, cols) ← transpose2DShape shape
  let outVals ← (List.range (rows * cols)).mapM fun flat => do
    let i := flat / rows
    let j := flat % rows
    getAt? values (j * cols + i)
  pure (.floats [cols, rows] outVals dtype)

private def transposeIf (tensor : TensorValue) (flag : Bool) : Except String TensorValue :=
  if !flag then
    pure tensor
  else
    match tensor.data with
    | .ints values => transposeInts tensor.shape values
    | .floats values => transposeFloats tensor.shape values tensor.dtype
    | .bools _ => .error "matrix transpose does not support bool tensors"

private def matMulInts (lhs rhs : TensorValue) : Except String TensorValue := do
  let lhsVals ← lhs.expectInts
  let rhsVals ← rhs.expectInts
  let (m, k1) ← transpose2DShape lhs.shape
  let (k2, n) ← transpose2DShape rhs.shape
  if k1 ≠ k2 then
    .error s!"matmul inner dimension mismatch: {k1} != {k2}"
  else
    let outVals ← (List.range (m * n)).mapM fun flat => do
      let i := flat / n
      let j := flat % n
      let partials ← (List.range k1).mapM fun k => do
        let a ← getAt? lhsVals (i * k1 + k)
        let b ← getAt? rhsVals (k * n + j)
        pure (a * b)
      pure (partials.foldl (· + ·) 0)
    pure (.ints [m, n] outVals)

private def matMulFloats (lhs rhs : TensorValue) : Except String TensorValue := do
  let lhsVals ← lhs.expectFloats
  let rhsVals ← rhs.expectFloats
  let (m, k1) ← transpose2DShape lhs.shape
  let (k2, n) ← transpose2DShape rhs.shape
  if k1 ≠ k2 then
    .error s!"matmul inner dimension mismatch: {k1} != {k2}"
  else
    let outVals ← (List.range (m * n)).mapM fun flat => do
      let i := flat / n
      let j := flat % n
      let partials ← (List.range k1).mapM fun k => do
        let a ← getAt? lhsVals (i * k1 + k)
        let b ← getAt? rhsVals (k * n + j)
        pure (a * b)
      pure (partials.foldl (· + ·) 0.0)
    pure (.floats [m, n] outVals lhs.dtype)

def evalMatMul (lhs rhs : TensorValue) : Except String TensorValue := do
  if lhs.dtype = .int64 && rhs.dtype = .int64 then
    matMulInts lhs rhs
  else if lhs.dtype.isFloat && rhs.dtype = lhs.dtype then
    matMulFloats lhs rhs
  else
    .error "matmul currently supports int64 or same float dtype matrices"

private def asBoolAttr (node : Node) (key : String) : Bool :=
  match AttrMap.find? node.attrs key with
  | some (.bool value) => value
  | some (.int value) => value ≠ 0
  | some (.nat value) => value ≠ 0
  | _ => false

private def asFloatAttr? (node : Node) (key : String) : Option Float :=
  match AttrMap.find? node.attrs key with
  | some (.float value) => some value
  | _ => none

private def addBiasInts (base bias : TensorValue) : Except String TensorValue := do
  let baseVals ← base.expectInts
  let biasVals ← bias.expectInts
  let [rows, cols] := base.shape
    | .error "gemm bias add expects rank-2 output"
  let outVals ← (List.range (rows * cols)).mapM fun flat => do
    let j := flat % cols
    let baseVal ← getAt? baseVals flat
    let biasVal ←
      match bias.shape with
      | [n] =>
          if n = cols then getAt? biasVals j else .error "1D gemm bias width mismatch"
      | [1, n] =>
          if n = cols then getAt? biasVals j else .error "2D gemm bias width mismatch"
      | [m, n] =>
          if m = rows && n = cols then getAt? biasVals flat else .error "2D gemm bias shape mismatch"
      | _ => .error s!"unsupported gemm bias shape {bias.shape}"
    pure (baseVal + biasVal)
  pure (.ints [rows, cols] outVals)

private def addBiasFloats (base bias : TensorValue) : Except String TensorValue := do
  let baseVals ← base.expectFloats
  let biasVals ← bias.expectFloats
  let [rows, cols] := base.shape
    | .error "gemm bias add expects rank-2 output"
  let outVals ← (List.range (rows * cols)).mapM fun flat => do
    let j := flat % cols
    let baseVal ← getAt? baseVals flat
    let biasVal ←
      match bias.shape with
      | [n] =>
          if n = cols then getAt? biasVals j else .error "1D gemm bias width mismatch"
      | [1, n] =>
          if n = cols then getAt? biasVals j else .error "2D gemm bias width mismatch"
      | [m, n] =>
          if m = rows && n = cols then getAt? biasVals flat else .error "2D gemm bias shape mismatch"
      | _ => .error s!"unsupported gemm bias shape {bias.shape}"
    pure (baseVal + biasVal)
  pure (.floats [rows, cols] outVals base.dtype)

def evalGemm (a b : TensorValue) (c? : Option TensorValue) (transA transB : Bool)
    (alpha beta : Float) : Except String TensorValue := do
  if !(alpha == 1.0 && beta == 1.0) then
    .error "gemm currently supports alpha=1 and beta=1 only"
  else
    let a' ← transposeIf a transA
    let b' ← transposeIf b transB
    let base ← evalMatMul a' b'
    match c? with
    | none => pure base
    | some c =>
        if base.dtype = .int64 && c.dtype = .int64 then
          addBiasInts base c
        else if base.dtype.isFloat && c.dtype = base.dtype then
          addBiasFloats base c
        else
          .error "gemm bias dtype mismatch"

def evalNode (node : Node) (env : Env) : Except String Env := do
  match node.op with
  | .matMul =>
      let _ ← requireArity node 2 1
      let lhs ← Env.lookup env node.inputs.head!
      let rhs ← Env.lookup env (node.inputs.getD 1 "")
      let output ← evalMatMul lhs rhs
      pure <| Env.insert env node.outputs.head! output
  | .gemm =>
      if node.inputs.length < 2 || node.inputs.length > 3 then
        .error s!"gemm expected 2 or 3 inputs, got {node.inputs.length}"
      else if node.outputs.length ≠ 1 then
        .error s!"gemm expected 1 output, got {node.outputs.length}"
      else
        let a ← Env.lookup env node.inputs.head!
        let b ← Env.lookup env (node.inputs.getD 1 "")
        let c? ←
          if node.inputs.length = 3 then
            pure (some (← Env.lookup env (node.inputs.getD 2 "")))
          else
            pure none
        let output ← evalGemm a b c? (asBoolAttr node "transA") (asBoolAttr node "transB")
          ((asFloatAttr? node "alpha").getD 1.0) ((asFloatAttr? node "beta").getD 1.0)
        pure <| Env.insert env node.outputs.head! output
  | _ => evalCoreNode node env

def evalGraph (graph : FragmentGraph) (inputs : Env) : Except String Env :=
  graph.nodes.foldlM (fun env node => evalNode node env) inputs

end HeytingLean.Bridge.Abbott.ONNX.EvalLinear
