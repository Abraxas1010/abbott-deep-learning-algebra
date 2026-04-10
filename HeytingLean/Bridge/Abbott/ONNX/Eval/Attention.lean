import HeytingLean.Bridge.Abbott.ONNX.Eval.Convolution

namespace HeytingLean.Bridge.Abbott.ONNX.EvalAttention

open HeytingLean.Bridge.Abbott.ONNX

private def getAt? (xs : List α) (idx : Nat) : Except String α :=
  match xs[idx]? with
  | some value => pure value
  | none => .error s!"index {idx} out of bounds"

private def flatIndex (shape : List Nat) (indices : List Nat) : Nat :=
  match shape, indices with
  | [], [] => 0
  | _ :: restShape, i :: restIdx =>
      let stride := elementCount restShape
      i * stride + flatIndex restShape restIdx
  | _, _ => 0

private def unravelIndex (shape : List Nat) (flat : Nat) : List Nat :=
  match shape with
  | [] => []
  | _ :: rest =>
      let stride := elementCount rest
      if stride = 0 then
        0 :: unravelIndex rest 0
      else
        (flat / stride) :: unravelIndex rest (flat % stride)

private def attrNat (node : Node) (key : String) (default : Nat) : Nat :=
  match AttrMap.find? node.attrs key with
  | some (.nat value) => value
  | some (.int value) => value.toNat
  | _ => default

private def evalSoftmax (tensor : TensorValue) (axis : Nat) : Except String TensorValue := do
  if !tensor.dtype.isFloat then
    .error "softmax currently supports float tensors only"
  else
    let values ← tensor.expectFloats
    let axis' ← normalizeAxis tensor.shape.length axis
    let outVals ← (List.range (elementCount tensor.shape)).mapM fun flat => do
      let outIdx := unravelIndex tensor.shape flat
      let axisLen := tensor.shape.getD axis' 0
      let lane ← (List.range axisLen).mapM fun slot => do
        let idx := (List.range tensor.shape.length).map fun i =>
          if i = axis' then slot else outIdx.getD i 0
        getAt? values (flatIndex tensor.shape idx)
      let exps := lane.map Float.exp
      let denom := exps.foldl (· + ·) 0.0
      let numer := exps.getD (outIdx.getD axis' 0) 0.0
      pure (numer / denom)
    pure (.floats tensor.shape outVals tensor.dtype)

private def attentionOutShape (qShape vShape : List Nat) : Except String (List Nat) := do
  match qShape, vShape with
  | [b, h, s, _], [b', h', _, dv] =>
      if b = b' && h = h' then pure [b, h, s, dv] else .error "attention batch/head mismatch"
  | [b, s, _], [b', _, dv] =>
      if b = b' then pure [b, s, dv] else .error "attention batch mismatch"
  | _, _ => .error "attention expects rank-3 or rank-4 q/k/v tensors"

private def headScore (qVals kVals : List Float) (qShape kShape : List Nat)
    (b h qPos kPos headDim : Nat) : Float :=
  let terms := (List.range headDim).map fun d =>
    let qIdx :=
      match qShape with
      | [_b, _h, _s, _d] => flatIndex qShape [b, h, qPos, d]
      | [_b, _s, _d] => flatIndex qShape [b, qPos, d]
      | _ => 0
    let kIdx :=
      match kShape with
      | [_b, _h, _s, _d] => flatIndex kShape [b, h, kPos, d]
      | [_b, _s, _d] => flatIndex kShape [b, kPos, d]
      | _ => 0
    qVals.getD qIdx 0.0 * kVals.getD kIdx 0.0
  let scale := Float.sqrt (Float.ofNat headDim)
  terms.foldl (· + ·) 0.0 / scale

private def valueAt (vals : List Float) (vShape : List Nat) (b h pos feat : Nat) : Float :=
  match vShape with
  | [_b, _h, _s, _d] => vals.getD (flatIndex vShape [b, h, pos, feat]) 0.0
  | [_b, _s, _d] => vals.getD (flatIndex vShape [b, pos, feat]) 0.0
  | _ => 0.0

private def evalAttention (node : Node) (q k v : TensorValue) : Except String TensorValue := do
  if !(q.dtype.isFloat && k.dtype = q.dtype && v.dtype = q.dtype) then
    .error "attention currently supports same float dtype q/k/v"
  else
    let qVals ← q.expectFloats
    let kVals ← k.expectFloats
    let vVals ← v.expectFloats
    let outShape ← attentionOutShape q.shape v.shape
    let outputRank := outShape.length
    let headCount :=
      match q.shape with
      | [_b, h, _s, _d] => h
      | [_b, _s, feature] =>
          let requested := attrNat node "num_heads" 1
          if requested = 0 || feature % requested ≠ 0 then 0 else requested
      | _ => 0
    if headCount = 0 then
      .error "attention requires rank-4 tensors or rank-3 tensors with divisible num_heads"
    else
      let seqK :=
        match k.shape with
        | [_b, _h, s, _d] => s
        | [_b, s, _d] => s
        | _ => 0
      let headDim :=
        match q.shape with
        | [_b, _h, _s, d] => d
        | [_b, _s, feature] => feature / headCount
        | _ => 0
      let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
        let outIdx := unravelIndex outShape flat
        let b := outIdx.getD 0 0
        let h := if outputRank = 4 then outIdx.getD 1 0 else 0
        let qPos := if outputRank = 4 then outIdx.getD 2 0 else outIdx.getD 1 0
        let feat := if outputRank = 4 then outIdx.getD 3 0 else outIdx.getD 2 0
        let scores := (List.range seqK).map fun kPos => headScore qVals kVals q.shape k.shape b h qPos kPos headDim
        let exps := scores.map Float.exp
        let denom := exps.foldl (· + ·) 0.0
        let weights := exps.map (· / denom)
        let acc := (List.range seqK).foldl
          (fun total kPos => total + weights.getD kPos 0.0 * valueAt vVals v.shape b h kPos feat)
          0.0
        pure acc
      pure (.floats outShape outVals q.dtype)

def evalNode (node : Node) (env : Env) : Except String Env := do
  match node.op with
  | .softmax =>
      let input ← Env.lookup env node.inputs.head!
      let output ← evalSoftmax input (attrNat node "axis" (input.shape.length - 1))
      pure <| Env.insert env node.outputs.head! output
  | .attention =>
      if node.inputs.length ≠ 3 then
        .error "attention expects q, k, v inputs"
      else
        let q ← Env.lookup env node.inputs.head!
        let k ← Env.lookup env (node.inputs.getD 1 "")
        let v ← Env.lookup env (node.inputs.getD 2 "")
        let output ← evalAttention node q k v
        pure <| Env.insert env node.outputs.head! output
  | _ => EvalConvolution.evalNode node env

def evalGraph (graph : FragmentGraph) (inputs : Env) : Except String Env :=
  graph.nodes.foldlM (fun env node => evalNode node env) inputs

end HeytingLean.Bridge.Abbott.ONNX.EvalAttention
