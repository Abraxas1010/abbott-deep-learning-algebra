import HeytingLean.Bridge.Abbott.ONNX.Eval.Linear

namespace HeytingLean.Bridge.Abbott.ONNX.EvalConvolution

open HeytingLean.Bridge.Abbott.ONNX

private def requireOutputs (node : Node) (count : Nat) : Except String Unit := do
  if node.outputs.length = count then
    pure ()
  else
    .error s!"{reprStr node.op} expected {count} outputs, got {node.outputs.length}"

private def getAt? (xs : List α) (idx : Nat) : Except String α :=
  match xs[idx]? with
  | some value => pure value
  | none => .error s!"index {idx} out of bounds"

private def flatIndex (shape : List Nat) (indices : List Nat) : Nat :=
  match shape, indices with
  | [], [] => 0
  | dim :: restShape, i :: restIdx =>
      let stride := elementCount restShape
      (min i dim) * stride + flatIndex restShape restIdx
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

private def attrFloat? (node : Node) (key : String) : Option Float :=
  match AttrMap.find? node.attrs key with
  | some (.float value) => some value
  | some (.int value) => some (Float.ofInt value)
  | some (.nat value) => some (Float.ofNat value)
  | _ => none

private def attrInt? (node : Node) (key : String) : Option Int :=
  match AttrMap.find? node.attrs key with
  | some (.int value) => some value
  | some (.nat value) => some (Int.ofNat value)
  | _ => none

private def attrBool (node : Node) (key : String) (default : Bool := false) : Bool :=
  match AttrMap.find? node.attrs key with
  | some (.bool value) => value
  | some (.int value) => value ≠ 0
  | some (.nat value) => value ≠ 0
  | _ => default

private def sourceIndexOfReduced (shape : List Nat) (axes : List Nat) (keepdims : Bool)
    (outIdx reducedIdx : List Nat) : List Nat :=
  let outAxisIdx (axis : Nat) : Nat :=
    if keepdims then
      outIdx.getD axis 0
    else
      let survivors := (List.range shape.length).filter fun i => !axes.contains i
      match survivors.findIdx? (· = axis) with
      | some pos => outIdx.getD pos 0
      | none => 0
  (List.range shape.length).map fun axis =>
    match axes.findIdx? (· = axis) with
    | some pos => reducedIdx.getD pos 0
    | none => outAxisIdx axis

private def reduceFloat (tensor : TensorValue) (axes : List Nat) (keepdims : Bool)
    (combine : Float -> Float -> Float) (init : Float) (finalize : Float -> Float) :
    Except String TensorValue := do
  let inputVals ← tensor.expectFloats
  let outShape ← reduceShape tensor.shape axes keepdims
  let axes' := if axes.isEmpty then List.range tensor.shape.length else axes
  let reducedShape := axes'.map fun axis => tensor.shape.getD axis 1
  let outVals ← (List.range (elementCount outShape)).mapM fun outFlat => do
    let outIdx := unravelIndex outShape outFlat
    let partials ← (List.range (elementCount reducedShape)).mapM fun redFlat => do
      let redIdx := unravelIndex reducedShape redFlat
      getAt? inputVals (flatIndex tensor.shape (sourceIndexOfReduced tensor.shape axes' keepdims outIdx redIdx))
    pure <| finalize (partials.foldl combine init)
  pure (.floats outShape outVals tensor.dtype)

private def reduceInt (tensor : TensorValue) (axes : List Nat) (keepdims : Bool)
    (combine : Int -> Int -> Int) (init : Int) (finalize : Int -> Int) :
    Except String TensorValue := do
  let inputVals ← tensor.expectInts
  let outShape ← reduceShape tensor.shape axes keepdims
  let axes' := if axes.isEmpty then List.range tensor.shape.length else axes
  let reducedShape := axes'.map fun axis => tensor.shape.getD axis 1
  let outVals ← (List.range (elementCount outShape)).mapM fun outFlat => do
    let outIdx := unravelIndex outShape outFlat
    let partials ← (List.range (elementCount reducedShape)).mapM fun redFlat => do
      let redIdx := unravelIndex reducedShape redFlat
      getAt? inputVals (flatIndex tensor.shape (sourceIndexOfReduced tensor.shape axes' keepdims outIdx redIdx))
    pure <| finalize (partials.foldl combine init)
  pure (.ints outShape outVals)

private def evalReduceSum (tensor : TensorValue) (axes : List Nat) (keepdims : Bool) :
    Except String TensorValue := do
  match tensor.data with
  | .ints _ => reduceInt tensor axes keepdims (· + ·) 0 id
  | .floats _ => reduceFloat tensor axes keepdims (· + ·) 0.0 id
  | .bools _ => .error "reduceSum does not support bool tensors"

private def evalReduceMean (tensor : TensorValue) (axes : List Nat) (keepdims : Bool) :
    Except String TensorValue := do
  match tensor.data with
  | .floats _ =>
      let axes' := if axes.isEmpty then List.range tensor.shape.length else axes
      let denom := Float.ofNat <| elementCount (axes'.map fun axis => tensor.shape.getD axis 1)
      reduceFloat tensor axes keepdims (· + ·) 0.0 (fun total => total / denom)
  | _ => .error "reduceMean currently supports float tensors only"

private def mapInts (tensor : TensorValue) (f : Int -> Int) : Except String TensorValue := do
  let values ← tensor.expectInts
  pure (.ints tensor.shape (values.map f))

private def mapFloats (tensor : TensorValue) (f : Float -> Float) : Except String TensorValue := do
  let values ← tensor.expectFloats
  pure (.floats tensor.shape (values.map f) tensor.dtype)

private def evalRelu (tensor : TensorValue) : Except String TensorValue := do
  match tensor.data with
  | .ints _ => mapInts tensor (fun x => max x 0)
  | .floats _ => mapFloats tensor (fun x => max x 0.0)
  | .bools _ => .error "relu does not support bool tensors"

private def evalClip (node : Node) (tensor : TensorValue) :
    Except String TensorValue := do
  match tensor.data with
  | .ints _ =>
      let lo := attrInt? node "min"
      let hi := attrInt? node "max"
      mapInts tensor fun x =>
        let x := match lo with | some lo => max x lo | none => x
        match hi with | some hi => min x hi | none => x
  | .floats _ =>
      let min? := attrFloat? node "min"
      let max? := attrFloat? node "max"
      mapFloats tensor fun x =>
        let x := match min? with | some lo => max x lo | none => x
        match max? with | some hi => min x hi | none => x
  | .bools _ => .error "clip does not support bool tensors"

private def padValueInt (node : Node) : Int :=
  match AttrMap.find? node.attrs "value" with
  | some (.int value) => value
  | _ => 0

private def padValueFloat (node : Node) : Float :=
  match AttrMap.find? node.attrs "value" with
  | some (.float value) => value
  | _ => 0.0

private def evalPad (node : Node) (tensor : TensorValue) : Except String TensorValue := do
  let mode :=
    match AttrMap.find? node.attrs "mode" with
    | some (.str value) => value
    | _ => "constant"
  if mode ≠ "constant" then
    .error s!"pad mode {mode} is unsupported"
  else
    let pads := node.attrNats? "pads" |>.getD []
    let outShape ← padShape tensor.shape pads
    let begins := pads.take tensor.shape.length
    let sourceIndex? (outIdx : List Nat) : Option (List Nat) :=
      let rec go : List Nat -> List Nat -> List Nat -> Option (List Nat)
        | [], [], acc => some acc.reverse
        | idx :: idxs, begin :: begins, acc =>
            if idx < begin then
              none
            else
              let src := idx - begin
              if src < tensor.shape.getD acc.length 0 then
                go idxs begins (src :: acc)
              else
                none
        | _, _, _ => none
      go outIdx begins []
    match tensor.data with
    | .ints values =>
        let padVal := padValueInt node
        let outVals := (List.range (elementCount outShape)).map fun flat =>
          let outIdx := unravelIndex outShape flat
          match sourceIndex? outIdx with
          | some src => values.getD (flatIndex tensor.shape src) padVal
          | none => padVal
        pure (.ints outShape outVals)
    | .floats values =>
        let padVal := padValueFloat node
        let outVals := (List.range (elementCount outShape)).map fun flat =>
          let outIdx := unravelIndex outShape flat
          match sourceIndex? outIdx with
          | some src => values.getD (flatIndex tensor.shape src) padVal
          | none => padVal
        pure (.floats outShape outVals tensor.dtype)
    | .bools values =>
        let outVals := (List.range (elementCount outShape)).map fun flat =>
          let outIdx := unravelIndex outShape flat
          match sourceIndex? outIdx with
          | some src => values.getD (flatIndex tensor.shape src) false
          | none => false
        pure (.bools outShape outVals)

private def convValueIndex (shape : List Nat) (n c h w : Nat) : Nat :=
  flatIndex shape [n, c, h, w]

private def evalConvInts (input weight : TensorValue) (bias? : Option TensorValue)
    (pads strides dilations : List Nat) : Except String TensorValue := do
  let inputVals ← input.expectInts
  let weightVals ← weight.expectInts
  let outShape ← convShape2D input.shape weight.shape pads strides dilations
  let kH := weight.shape.getD 2 0
  let kW := weight.shape.getD 3 0
  let strideH := strides.getD 0 1
  let strideW := strides.getD 1 1
  let padTop := pads.getD 0 0
  let padLeft := pads.getD 1 0
  let dilH := dilations.getD 0 1
  let dilW := dilations.getD 1 1
  let biasVals? ← match bias? with
    | none => pure none
    | some bias => pure (some (← bias.expectInts))
  let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
    let idx := unravelIndex outShape flat
    let n := idx.getD 0 0
    let oc := idx.getD 1 0
    let oh := idx.getD 2 0
    let ow := idx.getD 3 0
    let partials ← (List.range (weight.shape.getD 1 0)).mapM fun ic => do
      let terms ← (List.range (kH * kW)).mapM fun kernelFlat => do
        let kh := kernelFlat / kW
        let kw := kernelFlat % kW
        let inH := oh * strideH + kh * dilH
        let inW := ow * strideW + kw * dilW
        if inH < padTop || inW < padLeft then
          pure 0
        else
          let srcH := inH - padTop
          let srcW := inW - padLeft
          if srcH < input.shape.getD 2 0 && srcW < input.shape.getD 3 0 then
            let x := inputVals.getD (convValueIndex input.shape n ic srcH srcW) 0
            let w := weightVals.getD (convValueIndex weight.shape oc ic kh kw) 0
            pure (x * w)
          else
            pure 0
      pure (terms.foldl (· + ·) 0)
    let base := partials.foldl (· + ·) 0
    let biasTerm := match biasVals? with | some vals => vals.getD oc 0 | none => 0
    pure (base + biasTerm)
  pure (.ints outShape outVals)

private def evalConvFloats (input weight : TensorValue) (bias? : Option TensorValue)
    (pads strides dilations : List Nat) : Except String TensorValue := do
  let inputVals ← input.expectFloats
  let weightVals ← weight.expectFloats
  let outShape ← convShape2D input.shape weight.shape pads strides dilations
  let kH := weight.shape.getD 2 0
  let kW := weight.shape.getD 3 0
  let strideH := strides.getD 0 1
  let strideW := strides.getD 1 1
  let padTop := pads.getD 0 0
  let padLeft := pads.getD 1 0
  let dilH := dilations.getD 0 1
  let dilW := dilations.getD 1 1
  let biasVals? ← match bias? with
    | none => pure none
    | some bias => pure (some (← bias.expectFloats))
  let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
    let idx := unravelIndex outShape flat
    let n := idx.getD 0 0
    let oc := idx.getD 1 0
    let oh := idx.getD 2 0
    let ow := idx.getD 3 0
    let partials ← (List.range (weight.shape.getD 1 0)).mapM fun ic => do
      let terms ← (List.range (kH * kW)).mapM fun kernelFlat => do
        let kh := kernelFlat / kW
        let kw := kernelFlat % kW
        let inH := oh * strideH + kh * dilH
        let inW := ow * strideW + kw * dilW
        if inH < padTop || inW < padLeft then
          pure 0.0
        else
          let srcH := inH - padTop
          let srcW := inW - padLeft
          if srcH < input.shape.getD 2 0 && srcW < input.shape.getD 3 0 then
            let x := inputVals.getD (convValueIndex input.shape n ic srcH srcW) 0.0
            let w := weightVals.getD (convValueIndex weight.shape oc ic kh kw) 0.0
            pure (x * w)
          else
            pure 0.0
      pure (terms.foldl (· + ·) 0.0)
    let base := partials.foldl (· + ·) 0.0
    let biasTerm := match biasVals? with | some vals => vals.getD oc 0.0 | none => 0.0
    pure (base + biasTerm)
  pure (.floats outShape outVals input.dtype)

private def evalConv (input weight : TensorValue) (bias? : Option TensorValue)
    (pads strides dilations : List Nat) : Except String TensorValue := do
  if input.dtype = .int64 && weight.dtype = .int64 then
    evalConvInts input weight bias? pads strides dilations
  else if input.dtype.isFloat && weight.dtype = input.dtype then
    evalConvFloats input weight bias? pads strides dilations
  else
    .error "conv currently supports int64 or same float dtype inputs/weights"

private def evalBatchNorm (node : Node) (x scale bias mean variance : TensorValue) :
    Except String TensorValue := do
  if !(x.dtype.isFloat && scale.dtype = x.dtype && bias.dtype = x.dtype &&
      mean.dtype = x.dtype && variance.dtype = x.dtype) then
    .error "batchNormalization currently supports same float dtype inputs"
  else if x.shape.length ≠ 4 then
    .error "batchNormalization expects rank-4 NCHW input"
  else
    let xVals ← x.expectFloats
    let scaleVals ← scale.expectFloats
    let biasVals ← bias.expectFloats
    let meanVals ← mean.expectFloats
    let varVals ← variance.expectFloats
    let epsilon := (attrFloat? node "epsilon").getD 1e-5
    let outVals ← (List.range (elementCount x.shape)).mapM fun flat => do
      let idx := unravelIndex x.shape flat
      let c := idx.getD 1 0
      let xv ← getAt? xVals flat
      let γ := scaleVals.getD c 1.0
      let β := biasVals.getD c 0.0
      let μ := meanVals.getD c 0.0
      let σ2 := varVals.getD c 1.0
      pure (γ * ((xv - μ) / Float.sqrt (σ2 + epsilon)) + β)
    pure (.floats x.shape outVals x.dtype)

private def evalPoolInts (tensor : TensorValue) (kernelShape pads strides : List Nat)
    (useMax : Bool) : Except String TensorValue := do
  let values ← tensor.expectInts
  let outShape ← poolShape2D tensor.shape kernelShape pads strides
  let kH := kernelShape.getD 0 1
  let kW := kernelShape.getD 1 1
  let strideH := strides.getD 0 1
  let strideW := strides.getD 1 1
  let padTop := pads.getD 0 0
  let padLeft := pads.getD 1 0
  let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
    let idx := unravelIndex outShape flat
    let n := idx.getD 0 0
    let c := idx.getD 1 0
    let oh := idx.getD 2 0
    let ow := idx.getD 3 0
    let terms ← (List.range (kH * kW)).mapM fun kernelFlat => do
      let kh := kernelFlat / kW
      let kw := kernelFlat % kW
      let inH := oh * strideH + kh
      let inW := ow * strideW + kw
      if inH < padTop || inW < padLeft then
        pure none
      else
        let srcH := inH - padTop
        let srcW := inW - padLeft
        if srcH < tensor.shape.getD 2 0 && srcW < tensor.shape.getD 3 0 then
          pure (some (values.getD (convValueIndex tensor.shape n c srcH srcW) 0))
        else
          pure none
    let terms := terms.filterMap id
    if terms.isEmpty then
      pure 0
    else if useMax then
      pure (terms.foldl max terms.head!)
    else
      pure ((terms.foldl (· + ·) 0) / Int.ofNat terms.length)
  pure (.ints outShape outVals)

private def evalPoolFloats (tensor : TensorValue) (kernelShape pads strides : List Nat)
    (useMax : Bool) : Except String TensorValue := do
  let values ← tensor.expectFloats
  let outShape ← poolShape2D tensor.shape kernelShape pads strides
  let kH := kernelShape.getD 0 1
  let kW := kernelShape.getD 1 1
  let strideH := strides.getD 0 1
  let strideW := strides.getD 1 1
  let padTop := pads.getD 0 0
  let padLeft := pads.getD 1 0
  let outVals ← (List.range (elementCount outShape)).mapM fun flat => do
    let idx := unravelIndex outShape flat
    let n := idx.getD 0 0
    let c := idx.getD 1 0
    let oh := idx.getD 2 0
    let ow := idx.getD 3 0
    let terms ← (List.range (kH * kW)).mapM fun kernelFlat => do
      let kh := kernelFlat / kW
      let kw := kernelFlat % kW
      let inH := oh * strideH + kh
      let inW := ow * strideW + kw
      if inH < padTop || inW < padLeft then
        pure none
      else
        let srcH := inH - padTop
        let srcW := inW - padLeft
        if srcH < tensor.shape.getD 2 0 && srcW < tensor.shape.getD 3 0 then
          pure (some (values.getD (convValueIndex tensor.shape n c srcH srcW) 0.0))
        else
          pure none
    let terms := terms.filterMap id
    if terms.isEmpty then
      pure 0.0
    else if useMax then
      pure (terms.foldl max terms.head!)
    else
      pure ((terms.foldl (· + ·) 0.0) / Float.ofNat terms.length)
  pure (.floats outShape outVals tensor.dtype)

private def evalPool (tensor : TensorValue) (kernelShape pads strides : List Nat)
    (useMax : Bool) : Except String TensorValue := do
  match tensor.data with
  | .ints _ => evalPoolInts tensor kernelShape pads strides useMax
  | .floats _ => evalPoolFloats tensor kernelShape pads strides useMax
  | .bools _ => .error "pooling does not support bool tensors"

def evalNode (node : Node) (env : Env) : Except String Env := do
  match node.op with
  | .conv =>
      if node.inputs.length < 2 || node.inputs.length > 3 then
        .error s!"conv expected 2 or 3 inputs, got {node.inputs.length}"
      else
        let input ← Env.lookup env node.inputs.head!
        let weight ← Env.lookup env (node.inputs.getD 1 "")
        let bias? ←
          if node.inputs.length = 3 then
            pure (some (← Env.lookup env (node.inputs.getD 2 "")))
          else
            pure none
        let output ← evalConv input weight bias?
          (node.attrNats? "pads" |>.getD [0, 0, 0, 0])
          (node.attrNats? "strides" |>.getD [1, 1])
          (node.attrNats? "dilations" |>.getD [1, 1])
        pure <| Env.insert env node.outputs.head! output
  | .batchNormalization =>
      let _ ← requireOutputs node 1
      if node.inputs.length ≠ 5 then
        .error "batchNormalization expects 5 inputs"
      else
        let x ← Env.lookup env node.inputs.head!
        let scale ← Env.lookup env (node.inputs.getD 1 "")
        let bias ← Env.lookup env (node.inputs.getD 2 "")
        let mean ← Env.lookup env (node.inputs.getD 3 "")
        let variance ← Env.lookup env (node.inputs.getD 4 "")
        let output ← evalBatchNorm node x scale bias mean variance
        pure <| Env.insert env node.outputs.head! output
  | .relu =>
      let _ ← requireOutputs node 1
      let input ← Env.lookup env node.inputs.head!
      let output ← evalRelu input
      pure <| Env.insert env node.outputs.head! output
  | .clip =>
      let _ ← requireOutputs node 1
      let input ← Env.lookup env node.inputs.head!
      let output ← evalClip node input
      pure <| Env.insert env node.outputs.head! output
  | .reduceSum =>
      let _ ← requireOutputs node 1
      let input ← Env.lookup env node.inputs.head!
      let output ← evalReduceSum input (node.attrNats? "axes" |>.getD []) (attrBool node "keepdims" true)
      pure <| Env.insert env node.outputs.head! output
  | .reduceMean =>
      let _ ← requireOutputs node 1
      let input ← Env.lookup env node.inputs.head!
      let output ← evalReduceMean input (node.attrNats? "axes" |>.getD []) (attrBool node "keepdims" true)
      pure <| Env.insert env node.outputs.head! output
  | .maxPool =>
      let _ ← requireOutputs node 1
      let input ← Env.lookup env node.inputs.head!
      let output ← evalPool input (node.attrNats? "kernel_shape" |>.getD [1, 1])
        (node.attrNats? "pads" |>.getD [0, 0, 0, 0]) (node.attrNats? "strides" |>.getD [1, 1]) true
      pure <| Env.insert env node.outputs.head! output
  | .averagePool =>
      let _ ← requireOutputs node 1
      let input ← Env.lookup env node.inputs.head!
      let output ← evalPool input (node.attrNats? "kernel_shape" |>.getD [1, 1])
        (node.attrNats? "pads" |>.getD [0, 0, 0, 0]) (node.attrNats? "strides" |>.getD [1, 1]) false
      pure <| Env.insert env node.outputs.head! output
  | .pad =>
      let _ ← requireOutputs node 1
      let input ← Env.lookup env node.inputs.head!
      let output ← evalPad node input
      pure <| Env.insert env node.outputs.head! output
  | _ => EvalLinear.evalNode node env

def evalGraph (graph : FragmentGraph) (inputs : Env) : Except String Env :=
  graph.nodes.foldlM (fun env node => evalNode node env) inputs

end HeytingLean.Bridge.Abbott.ONNX.EvalConvolution
