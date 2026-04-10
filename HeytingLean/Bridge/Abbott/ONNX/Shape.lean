import HeytingLean.Bridge.Abbott.ONNX.Types

namespace HeytingLean.Bridge.Abbott.ONNX

def elementCount (dims : List Nat) : Nat :=
  dims.foldl (· * ·) 1

def normalizeAxis (rank axis : Nat) : Except String Nat :=
  if axis < rank then
    pure axis
  else
    .error s!"axis {axis} out of bounds for rank {rank}"

private def broadcastRev : List Nat -> List Nat -> Except String (List Nat)
  | [], ys => pure ys
  | xs, [] => pure xs
  | x :: xs, y :: ys => do
      let tail ← broadcastRev xs ys
      if x = y then
        pure (x :: tail)
      else if x = 1 then
        pure (y :: tail)
      else if y = 1 then
        pure (x :: tail)
      else
        .error s!"broadcast mismatch: {x} versus {y}"

def broadcastShape (lhs rhs : List Nat) : Except String (List Nat) := do
  let merged ← broadcastRev lhs.reverse rhs.reverse
  pure merged.reverse

def broadcastShapes (shapes : List (List Nat)) : Except String (List Nat) := do
  match shapes with
  | [] => pure []
  | first :: rest => rest.foldlM broadcastShape first

def reshapeCompatible (src dst : List Nat) : Bool :=
  elementCount src = elementCount dst

def reshapeShape (src dst : List Nat) : Except String (List Nat) :=
  if reshapeCompatible src dst then
    pure dst
  else
    .error s!"reshape mismatch: {src} -> {dst}"

def transposeShape (shape perm : List Nat) : Except String (List Nat) := do
  if perm.length ≠ shape.length then
    .error s!"transpose permutation rank mismatch: {perm.length} != {shape.length}"
  else if perm.eraseDups.length ≠ perm.length then
    .error "transpose permutation contains duplicates"
  else if perm.all (fun axis => axis < shape.length) then
    pure <| perm.map fun axis => shape.getD axis 0
  else
    .error "transpose permutation contains out-of-bounds axis"

def unsqueezeShape (shape axes : List Nat) : Except String (List Nat) :=
  axes.foldlM
    (fun acc axis => do
      if axis ≤ acc.length then
        pure (acc.take axis ++ [1] ++ acc.drop axis)
      else
        .error s!"unsqueeze axis {axis} out of bounds for rank {acc.length}")
    shape

def squeezeShape (shape axes : List Nat) : Except String (List Nat) := do
  axes.foldlM
    (fun acc axis => do
      let axis' ← normalizeAxis acc.length axis
      match acc[axis']? with
      | some 1 => pure (acc.take axis' ++ acc.drop (axis' + 1))
      | some dim => .error s!"cannot squeeze axis {axis'} with size {dim}"
      | none => .error s!"axis {axis'} out of bounds")
    shape

def flattenShape (shape : List Nat) (axis : Nat := 1) : Except String (List Nat) := do
  if axis > shape.length then
    .error s!"flatten axis {axis} out of bounds for rank {shape.length}"
  else
    let left := elementCount (shape.take axis)
    let right := elementCount (shape.drop axis)
    pure [left, right]

def concatShape (axis : Nat) (shapes : List (List Nat)) : Except String (List Nat) := do
  let first :: rest := shapes
    | .error "concat requires at least one shape"
  let axis' ← normalizeAxis first.length axis
  let mergedAxis ← rest.foldlM
    (fun acc shape => do
      if shape.length ≠ first.length then
        .error "concat rank mismatch"
      else if (List.range first.length).all (fun i => i = axis' || first.getD i 0 = shape.getD i 0) then
        pure (acc + shape.getD axis' 0)
      else
        .error "concat non-axis dimensions mismatch")
    (first.getD axis' 0)
  pure <| first.take axis' ++ [mergedAxis] ++ first.drop (axis' + 1)

def whereShape (cond lhs rhs : List Nat) : Except String (List Nat) := do
  broadcastShapes [cond, lhs, rhs]

private def normalizeAxisList (rank : Nat) (axes : List Nat) : Except String (List Nat) := do
  axes.mapM (normalizeAxis rank)

private def sliceLen (start stop step : Nat) : Nat :=
  if start < stop then
    ((stop - start - 1) / step) + 1
  else
    0

private def replaceAt : List Nat -> Nat -> Nat -> List Nat
  | [], _, _ => []
  | _ :: rest, 0, value => value :: rest
  | head :: rest, i + 1, value => head :: replaceAt rest i value

def sliceShape (shape starts ends axes steps : List Nat) : Except String (List Nat) := do
  if starts.length ≠ ends.length then
    .error "slice starts/ends length mismatch"
  else if starts.length ≠ axes.length then
    .error "slice starts/axes length mismatch"
  else if starts.length ≠ steps.length then
    .error "slice starts/steps length mismatch"
  else
    let axes' ← normalizeAxisList shape.length axes
    let rec applyUpdates (current : List Nat) :
        List Nat -> List Nat -> List Nat -> List Nat -> Except String (List Nat)
      | [], [], [], [] => pure current
      | start :: restStarts, stop :: restEnds, axis :: restAxes, step :: restSteps => do
          if step = 0 then
            .error "slice step cannot be zero"
          else
            match current[axis]? with
            | none => .error s!"slice axis {axis} out of bounds"
            | some dim =>
                let clampedStart := min start dim
                let clampedStop := min stop dim
                let newDim := sliceLen clampedStart clampedStop step
                let next := replaceAt current axis newDim
                applyUpdates next restStarts restEnds restAxes restSteps
      | _, _, _, _ => .error "slice attribute mismatch"
    applyUpdates shape starts ends axes' steps

def gatherShape (dataShape indicesShape : List Nat) (axis : Nat) : Except String (List Nat) := do
  let axis' ← normalizeAxis dataShape.length axis
  pure <| dataShape.take axis' ++ indicesShape ++ dataShape.drop (axis' + 1)

def reduceShape (shape axes : List Nat) (keepdims : Bool) : Except String (List Nat) := do
  let axes' :=
    if axes.isEmpty then
      List.range shape.length
    else
      axes
  let axes'' ← normalizeAxisList shape.length axes'
  if keepdims then
    pure <|
      (List.range shape.length).map fun i =>
        if axes''.contains i then 1 else shape.getD i 1
  else
    pure <|
      (List.range shape.length).filterMap fun i =>
        if axes''.contains i then none else some (shape.getD i 1)

def padShape (shape pads : List Nat) : Except String (List Nat) := do
  if pads.length ≠ 2 * shape.length then
    .error s!"pad attribute length {pads.length} does not match rank {shape.length}"
  else
    let begins := pads.take shape.length
    let ends := pads.drop shape.length
    pure <| (List.zipWith (· + ·) shape begins |>.zipWith (· + ·) ends)

def windowOutputLength (input kernel stride padLo padHi dilation : Nat) : Except String Nat := do
  if stride = 0 then
    .error "stride cannot be zero"
  else
    let receptive := dilation * (kernel - 1) + 1
    let padded := input + padLo + padHi
    if padded + 1 < receptive + 1 then
      pure 0
    else
      pure <| ((padded - receptive) / stride) + 1

def convShape2D (inputShape weightShape : List Nat) (pads strides dilations : List Nat) :
    Except String (List Nat) := do
  if inputShape.length ≠ 4 then
    .error "conv expects rank-4 NCHW input"
  else if weightShape.length ≠ 4 then
    .error "conv expects rank-4 OIHW weights"
  else if pads.length ≠ 4 then
    .error "conv expects 4 pad entries"
  else if strides.length ≠ 2 then
    .error "conv expects 2 stride entries"
  else if dilations.length ≠ 2 then
    .error "conv expects 2 dilation entries"
  else if inputShape.getD 1 0 ≠ weightShape.getD 1 0 then
    .error "conv channel mismatch"
  else
    let outH ← windowOutputLength (inputShape.getD 2 0) (weightShape.getD 2 0)
      (strides.getD 0 1) (pads.getD 0 0) (pads.getD 2 0) (dilations.getD 0 1)
    let outW ← windowOutputLength (inputShape.getD 3 0) (weightShape.getD 3 0)
      (strides.getD 1 1) (pads.getD 1 0) (pads.getD 3 0) (dilations.getD 1 1)
    pure [inputShape.getD 0 0, weightShape.getD 0 0, outH, outW]

def poolShape2D (inputShape kernelShape pads strides : List Nat) : Except String (List Nat) := do
  if inputShape.length ≠ 4 then
    .error "pool expects rank-4 NCHW input"
  else if kernelShape.length ≠ 2 then
    .error "pool expects 2 kernel dimensions"
  else if pads.length ≠ 4 then
    .error "pool expects 4 pad entries"
  else if strides.length ≠ 2 then
    .error "pool expects 2 stride entries"
  else
    let outH ← windowOutputLength (inputShape.getD 2 0) (kernelShape.getD 0 1)
      (strides.getD 0 1) (pads.getD 0 0) (pads.getD 2 0) 1
    let outW ← windowOutputLength (inputShape.getD 3 0) (kernelShape.getD 1 1)
      (strides.getD 1 1) (pads.getD 1 0) (pads.getD 3 0) 1
    pure [inputShape.getD 0 0, inputShape.getD 1 0, outH, outW]

end HeytingLean.Bridge.Abbott.ONNX
