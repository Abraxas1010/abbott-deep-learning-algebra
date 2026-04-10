import HeytingLean.Bridge.Abbott.ONNX.TensorValue

namespace HeytingLean.Bridge.Abbott.ONNX

abbrev Env := List (String × TensorValue)

namespace Env

def lookup (env : Env) (name : String) : Except String TensorValue :=
  match env.findSome? (fun entry => if entry.fst = name then some entry.snd else none) with
  | some value => pure value
  | none => .error s!"missing tensor binding: {name}"

def insert (env : Env) (name : String) (value : TensorValue) : Env :=
  (name, value) :: env.filter (fun entry => entry.fst ≠ name)

end Env

namespace AttrValue

def asNat? : AttrValue → Option Nat
  | .nat n => some n
  | .int z => Int.toNat? z
  | _ => none

def asNats? : AttrValue → Option (List Nat)
  | .nats ns => some ns
  | .ints zs => zs.mapM Int.toNat?
  | _ => none

end AttrValue

namespace Node

def attrNat? (node : Node) (key : String) : Option Nat :=
  (AttrMap.find? node.attrs key).bind AttrValue.asNat?

def attrNats? (node : Node) (key : String) : Option (List Nat) :=
  (AttrMap.find? node.attrs key).bind AttrValue.asNats?

def attrShape? (node : Node) (key : String) : Option ShapeExpr :=
  match AttrMap.find? node.attrs key with
  | some (.shape shape) => some shape
  | _ => none

def attrDType? (node : Node) (key : String) : Option DType :=
  match AttrMap.find? node.attrs key with
  | some (.dtype dtype) => some dtype
  | _ => none

def attrInt? (node : Node) (key : String) : Option Int :=
  match AttrMap.find? node.attrs key with
  | some (.int value) => some value
  | _ => none

end Node

private def requireArity (node : Node) (inputs outputs : Nat) : Except String Unit := do
  if node.inputs.length ≠ inputs then
    .error s!"{reprStr node.op} expected {inputs} inputs, got {node.inputs.length}"
  else if node.outputs.length ≠ outputs then
    .error s!"{reprStr node.op} expected {outputs} outputs, got {node.outputs.length}"
  else
    pure ()

private def flatIndex (shape : List Nat) (indices : List Nat) : Nat :=
  match shape, indices with
  | [], [] => 0
  | _dim :: restShape, i :: restIdx =>
      let stride := elementCount restShape
      i * stride + flatIndex restShape restIdx
  | _, _ => 0

private def unravelIndex (shape : List Nat) (flat : Nat) : List Nat :=
  match shape with
  | [] => []
  | _dim :: rest =>
      let stride := elementCount rest
      if stride = 0 then
        0 :: unravelIndex rest 0
      else
        (flat / stride) :: unravelIndex rest (flat % stride)

private def getAt? (xs : List α) (idx : Nat) : Except String α :=
  match xs[idx]? with
  | some value => pure value
  | none => .error s!"index {idx} out of bounds"

private def findIndex? (xs : List Nat) (target : Nat) : Option Nat :=
  let rec go (i : Nat) : List Nat -> Option Nat
    | [] => none
    | x :: rest => if x = target then some i else go (i + 1) rest
  go 0 xs

private def inversePerm (perm : List Nat) : Except String (List Nat) :=
  (List.range perm.length).mapM fun axis =>
    match findIndex? perm axis with
    | some outAxis => pure outAxis
    | none => .error "invalid permutation"

private def broadcastSourceIndex (inputShape outputShape outIndex : List Nat) : List Nat :=
  let pad := outputShape.length - inputShape.length
  let padded := List.replicate pad 1 ++ inputShape
  let aligned :=
    List.zip padded outIndex |>.map fun (dim, idx) => if dim = 1 then 0 else idx
  aligned.drop pad

private def mapTensorDataM (tensor : TensorValue) (fInt : List Int -> Except String TensorValue)
    (fFloat : List Float -> Except String TensorValue)
    (fBool : List Bool -> Except String TensorValue) : Except String TensorValue :=
  match tensor.data with
  | .ints values => fInt values
  | .floats values => fFloat values
  | .bools values => fBool values

private def transposeData (tensor : TensorValue) (perm : List Nat) : Except String TensorValue := do
  let outShape ← transposeShape tensor.shape perm
  let invPerm ← inversePerm perm
  mapTensorDataM tensor
    (fun values => do
      let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
        let outIdx := unravelIndex outShape flat
        let srcIdx := invPerm.map fun outAxis => outIdx.getD outAxis 0
        getAt? values (flatIndex tensor.shape srcIdx)
      pure (.ints outShape outVals))
    (fun values => do
      let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
        let outIdx := unravelIndex outShape flat
        let srcIdx := invPerm.map fun outAxis => outIdx.getD outAxis 0
        getAt? values (flatIndex tensor.shape srcIdx)
      pure (.floats outShape outVals tensor.dtype))
    (fun values => do
      let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
        let outIdx := unravelIndex outShape flat
        let srcIdx := invPerm.map fun outAxis => outIdx.getD outAxis 0
        getAt? values (flatIndex tensor.shape srcIdx)
      pure (.bools outShape outVals))

private def expandData (tensor : TensorValue) (target : List Nat) : Except String TensorValue := do
  let outShape ← broadcastShape tensor.shape target
  if outShape ≠ target then
    .error s!"expand target {target} is not reachable from input shape {tensor.shape}"
  else
    mapTensorDataM tensor
      (fun values => do
        let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
          let outIdx := unravelIndex outShape flat
          let srcIdx := broadcastSourceIndex tensor.shape outShape outIdx
          getAt? values (flatIndex tensor.shape srcIdx)
        pure (.ints outShape outVals))
      (fun values => do
        let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
          let outIdx := unravelIndex outShape flat
          let srcIdx := broadcastSourceIndex tensor.shape outShape outIdx
          getAt? values (flatIndex tensor.shape srcIdx)
        pure (.floats outShape outVals tensor.dtype))
      (fun values => do
        let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
          let outIdx := unravelIndex outShape flat
          let srcIdx := broadcastSourceIndex tensor.shape outShape outIdx
          getAt? values (flatIndex tensor.shape srcIdx)
        pure (.bools outShape outVals))

private def binaryElementwiseInt (lhs rhs : TensorValue) (op : Int -> Int -> Int) :
    Except String TensorValue := do
  let outShape ← broadcastShape lhs.shape rhs.shape
  let lhsVals ← lhs.expectInts
  let rhsVals ← rhs.expectInts
  let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
    let outIdx := unravelIndex outShape flat
    let lhsIdx := broadcastSourceIndex lhs.shape outShape outIdx
    let rhsIdx := broadcastSourceIndex rhs.shape outShape outIdx
    pure <| op (lhsVals.getD (flatIndex lhs.shape lhsIdx) 0) (rhsVals.getD (flatIndex rhs.shape rhsIdx) 0)
  pure (.ints outShape outVals)

private def binaryElementwiseFloat (lhs rhs : TensorValue) (op : Float -> Float -> Float) :
    Except String TensorValue := do
  let outShape ← broadcastShape lhs.shape rhs.shape
  let lhsVals ← lhs.expectFloats
  let rhsVals ← rhs.expectFloats
  let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
    let outIdx := unravelIndex outShape flat
    let lhsIdx := broadcastSourceIndex lhs.shape outShape outIdx
    let rhsIdx := broadcastSourceIndex rhs.shape outShape outIdx
    pure <| op (lhsVals.getD (flatIndex lhs.shape lhsIdx) 0.0) (rhsVals.getD (flatIndex rhs.shape rhsIdx) 0.0)
  pure (.floats outShape outVals lhs.dtype)

private def addData (lhs rhs : TensorValue) : Except String TensorValue := do
  if lhs.dtype = .int64 && rhs.dtype = .int64 then
    binaryElementwiseInt lhs rhs (· + ·)
  else if lhs.dtype.isFloat && rhs.dtype = lhs.dtype then
    binaryElementwiseFloat lhs rhs (· + ·)
  else
    .error "add currently supports int64 or same float dtype"

private def mulData (lhs rhs : TensorValue) : Except String TensorValue := do
  if lhs.dtype = .int64 && rhs.dtype = .int64 then
    binaryElementwiseInt lhs rhs (· * ·)
  else if lhs.dtype.isFloat && rhs.dtype = lhs.dtype then
    binaryElementwiseFloat lhs rhs (· * ·)
  else
    .error "mul currently supports int64 or same float dtype"

private def whereData (cond lhs rhs : TensorValue) : Except String TensorValue := do
  let outShape ← whereShape cond.shape lhs.shape rhs.shape
  let condVals ← cond.expectBools
  match lhs.data, rhs.data with
  | .ints lhsVals, .ints rhsVals =>
      let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
        let outIdx := unravelIndex outShape flat
        let condIdx := broadcastSourceIndex cond.shape outShape outIdx
        let lhsIdx := broadcastSourceIndex lhs.shape outShape outIdx
        let rhsIdx := broadcastSourceIndex rhs.shape outShape outIdx
        let flag := condVals.getD (flatIndex cond.shape condIdx) false
        pure <| if flag then lhsVals.getD (flatIndex lhs.shape lhsIdx) 0 else rhsVals.getD (flatIndex rhs.shape rhsIdx) 0
      pure (.ints outShape outVals)
  | .bools lhsVals, .bools rhsVals =>
      let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
        let outIdx := unravelIndex outShape flat
        let condIdx := broadcastSourceIndex cond.shape outShape outIdx
        let lhsIdx := broadcastSourceIndex lhs.shape outShape outIdx
        let rhsIdx := broadcastSourceIndex rhs.shape outShape outIdx
        let flag := condVals.getD (flatIndex cond.shape condIdx) false
        pure <| if flag then lhsVals.getD (flatIndex lhs.shape lhsIdx) false else rhsVals.getD (flatIndex rhs.shape rhsIdx) false
      pure (.bools outShape outVals)
  | _, _ => .error "where currently supports matching int64 or bool payloads"

private def concatData (axis : Nat) (tensors : List TensorValue) : Except String TensorValue := do
  let first :: _rest := tensors
    | .error "concat requires at least one tensor"
  let outShape ← concatShape axis (tensors.map (·.shape))
  let axis' ← normalizeAxis first.shape.length axis
  let outer := elementCount (first.shape.take axis')
  let inner := elementCount (first.shape.drop (axis' + 1))
  let axisSizes := tensors.map fun tensor => tensor.shape.getD axis' 0
  match first.data with
  | .ints _ =>
      let payloads ← tensors.mapM TensorValue.expectInts
      let outVals ← (List.range outer).foldlM
        (fun acc outerIdx => do
          let parts ← (List.zip axisSizes payloads).mapM fun entry => do
            let axisSize := entry.fst
            let values := entry.snd
            let start := outerIdx * axisSize * inner
            pure ((values.drop start).take (axisSize * inner))
          pure (acc ++ parts.foldr List.append []))
        []
      pure (.ints outShape outVals)
  | .bools _ =>
      let payloads ← tensors.mapM TensorValue.expectBools
      let outVals ← (List.range outer).foldlM
        (fun acc outerIdx => do
          let parts ← (List.zip axisSizes payloads).mapM fun entry => do
            let axisSize := entry.fst
            let values := entry.snd
            let start := outerIdx * axisSize * inner
            pure ((values.drop start).take (axisSize * inner))
          pure (acc ++ parts.foldr List.append []))
        []
      pure (.bools outShape outVals)
  | .floats _ =>
      let payloads ← tensors.mapM TensorValue.expectFloats
      let outVals ← (List.range outer).foldlM
        (fun acc outerIdx => do
          let parts ← (List.zip axisSizes payloads).mapM fun entry => do
            let axisSize := entry.fst
            let values := entry.snd
            let start := outerIdx * axisSize * inner
            pure ((values.drop start).take (axisSize * inner))
          pure (acc ++ parts.foldr List.append []))
        []
      pure (.floats outShape outVals first.dtype)

private def sliceData (tensor : TensorValue) (starts ends axes steps : List Nat) :
    Except String TensorValue := do
  let axes' := if axes.isEmpty then List.range starts.length else axes
  let steps' := if steps.isEmpty then List.replicate starts.length 1 else steps
  let outShape ← sliceShape tensor.shape starts ends axes' steps'
  let axisStartMap := List.zip axes' starts
  let axisStepMap := List.zip axes' steps'
  let sourceIndex (outIdx : List Nat) : List Nat :=
    (List.range tensor.shape.length).map fun axis =>
      let start := (axisStartMap.findSome? fun entry =>
        if entry.fst = axis then some entry.snd else none).getD 0
      let step := (axisStepMap.findSome? fun entry =>
        if entry.fst = axis then some entry.snd else none).getD 1
      start + step * outIdx.getD axis 0
  mapTensorDataM tensor
    (fun values => do
      let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
        let outIdx := unravelIndex outShape flat
        getAt? values (flatIndex tensor.shape (sourceIndex outIdx))
      pure (.ints outShape outVals))
    (fun values => do
      let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
        let outIdx := unravelIndex outShape flat
        getAt? values (flatIndex tensor.shape (sourceIndex outIdx))
      pure (.floats outShape outVals tensor.dtype))
    (fun values => do
      let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
        let outIdx := unravelIndex outShape flat
        getAt? values (flatIndex tensor.shape (sourceIndex outIdx))
      pure (.bools outShape outVals))

private def gatherSourceIndex (indicesShape outIdx : List Nat) (axis gatherIdx : Nat) : List Nat :=
  outIdx.take axis ++ [gatherIdx] ++ outIdx.drop (axis + indicesShape.length)

private def gatherData (data indices : TensorValue) (axis : Nat) : Except String TensorValue := do
  let axis' ← normalizeAxis data.shape.length axis
  let indexVals ← indices.expectInts
  let outShape ← gatherShape data.shape indices.shape axis'
  let gatherAt (outIdx : List Nat) : Except String Nat := do
    let indexIdx := (outIdx.drop axis').take indices.shape.length
    let raw ← getAt? indexVals (flatIndex indices.shape indexIdx)
    match Int.toNat? raw with
    | some n =>
        if n < data.shape.getD axis' 0 then
          pure n
        else
          .error s!"gather index {n} out of bounds for axis {axis'}"
    | none => .error s!"gather index must be nonnegative, got {raw}"
  mapTensorDataM data
    (fun values => do
      let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
        let outIdx := unravelIndex outShape flat
        let gatherIdx ← gatherAt outIdx
        getAt? values (flatIndex data.shape (gatherSourceIndex indices.shape outIdx axis' gatherIdx))
      pure (.ints outShape outVals))
    (fun values => do
      let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
        let outIdx := unravelIndex outShape flat
        let gatherIdx ← gatherAt outIdx
        getAt? values (flatIndex data.shape (gatherSourceIndex indices.shape outIdx axis' gatherIdx))
      pure (.floats outShape outVals data.dtype))
    (fun values => do
      let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
        let outIdx := unravelIndex outShape flat
        let gatherIdx ← gatherAt outIdx
        getAt? values (flatIndex data.shape (gatherSourceIndex indices.shape outIdx axis' gatherIdx))
      pure (.bools outShape outVals))

def evalCoreNode (node : Node) (env : Env) : Except String Env := do
  match node.op with
  | .identity =>
      let _ ← requireArity node 1 1
      let input ← Env.lookup env node.inputs.head!
      pure <| Env.insert env node.outputs.head! input
  | .shape =>
      let _ ← requireArity node 1 1
      let input ← Env.lookup env node.inputs.head!
      pure <| Env.insert env node.outputs.head! input.shapeTensor
  | .constantOfShape =>
      let _ ← requireArity node 1 1
      let shapeTensor ← Env.lookup env node.inputs.head!
      let dims ← shapeTensor.expectInts
      let shape ← dims.mapM fun dim =>
        match Int.toNat? dim with
        | some n => pure n
        | none => .error s!"shape dimension must be nonnegative, got {dim}"
      let value := node.attrInt? "value" |>.getD 0
      pure <| Env.insert env node.outputs.head! (.constantOfShapeInts shape value)
  | .cast =>
      let _ ← requireArity node 1 1
      let input ← Env.lookup env node.inputs.head!
      let target := node.attrDType? "to" |>.getD .int64
      let output ← input.cast target
      pure <| Env.insert env node.outputs.head! output
  | .reshape =>
      let _ ← requireArity node 1 1
      let input ← Env.lookup env node.inputs.head!
      let targetShape ←
        match node.attrShape? "shape" with
        | some shape =>
            match shape.knownDims? with
            | some dims => pure dims
            | none => .error "reshape target must be fully static in tranche 1"
        | none => .error "reshape requires `shape` attribute"
      let output ← input.reshape targetShape
      pure <| Env.insert env node.outputs.head! output
  | .flatten =>
      let _ ← requireArity node 1 1
      let input ← Env.lookup env node.inputs.head!
      let axis := node.attrNat? "axis" |>.getD 1
      let targetShape ← flattenShape input.shape axis
      let output ← input.withShape targetShape
      pure <| Env.insert env node.outputs.head! output
  | .unsqueeze =>
      let _ ← requireArity node 1 1
      let input ← Env.lookup env node.inputs.head!
      let axes := node.attrNats? "axes" |>.getD []
      let targetShape ← unsqueezeShape input.shape axes
      let output ← input.withShape targetShape
      pure <| Env.insert env node.outputs.head! output
  | .squeeze =>
      let _ ← requireArity node 1 1
      let input ← Env.lookup env node.inputs.head!
      let axes := node.attrNats? "axes" |>.getD []
      let targetShape ← squeezeShape input.shape axes
      let output ← input.withShape targetShape
      pure <| Env.insert env node.outputs.head! output
  | .transpose =>
      let _ ← requireArity node 1 1
      let input ← Env.lookup env node.inputs.head!
      let perm := node.attrNats? "perm" |>.getD (List.range input.shape.length)
      let output ← transposeData input perm
      pure <| Env.insert env node.outputs.head! output
  | .expand =>
      let _ ← requireArity node 1 1
      let input ← Env.lookup env node.inputs.head!
      let targetShape ←
        match node.attrShape? "shape" with
        | some shape =>
            match shape.knownDims? with
            | some dims => pure dims
            | none => .error "expand target must be fully static in tranche 1"
        | none => .error "expand requires `shape` attribute"
      let output ← expandData input targetShape
      pure <| Env.insert env node.outputs.head! output
  | .add =>
      let _ ← requireArity node 2 1
      let lhs ← Env.lookup env node.inputs.head!
      let rhs ← Env.lookup env (node.inputs.getD 1 "")
      let output ← addData lhs rhs
      pure <| Env.insert env node.outputs.head! output
  | .mul =>
      let _ ← requireArity node 2 1
      let lhs ← Env.lookup env node.inputs.head!
      let rhs ← Env.lookup env (node.inputs.getD 1 "")
      let output ← mulData lhs rhs
      pure <| Env.insert env node.outputs.head! output
  | .where =>
      let _ ← requireArity node 3 1
      let cond ← Env.lookup env node.inputs.head!
      let lhs ← Env.lookup env (node.inputs.getD 1 "")
      let rhs ← Env.lookup env (node.inputs.getD 2 "")
      let output ← whereData cond lhs rhs
      pure <| Env.insert env node.outputs.head! output
  | .concat =>
      if node.outputs.length ≠ 1 then
        .error "concat expects exactly one output"
      else
        let inputs ← node.inputs.mapM (Env.lookup env)
        let axis := node.attrNat? "axis" |>.getD 0
        let output ← concatData axis inputs
        pure <| Env.insert env node.outputs.head! output
  | .slice =>
      let _ ← requireArity node 1 1
      let input ← Env.lookup env node.inputs.head!
      let starts := node.attrNats? "starts" |>.getD []
      let ends := node.attrNats? "ends" |>.getD []
      let axes := node.attrNats? "axes" |>.getD []
      let steps := node.attrNats? "steps" |>.getD []
      let output ← sliceData input starts ends axes steps
      pure <| Env.insert env node.outputs.head! output
  | .gather =>
      let _ ← requireArity node 2 1
      let data ← Env.lookup env node.inputs.head!
      let indices ← Env.lookup env (node.inputs.getD 1 "")
      let axis := node.attrNat? "axis" |>.getD 0
      let output ← gatherData data indices axis
      pure <| Env.insert env node.outputs.head! output
  | _ =>
      .error s!"core evaluator does not support {reprStr node.op} yet"

def evalGraph (graph : FragmentGraph) (inputs : Env) : Except String Env :=
  graph.nodes.foldlM (fun env node => evalCoreNode node env) inputs

end HeytingLean.Bridge.Abbott.ONNX
