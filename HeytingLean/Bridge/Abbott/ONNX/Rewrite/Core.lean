import HeytingLean.Bridge.Abbott.ONNX.Eval.Attention

namespace HeytingLean.Bridge.Abbott.ONNX.RewriteCore

open HeytingLean.Bridge.Abbott.ONNX

private def attrBool (node : Node) (key : String) : Bool :=
  match AttrMap.find? node.attrs key with
  | some (.bool value) => value
  | some (.int value) => value ≠ 0
  | some (.nat value) => value ≠ 0
  | _ => false

def evalOutput (graph : FragmentGraph) (inputs : Env) (name : String := "out") : Except String TensorValue := do
  let env ← EvalAttention.evalGraph graph inputs
  Env.lookup env name

def outputsAgree (before after : FragmentGraph) (inputs : Env) (name : String := "out") : Bool :=
  match evalOutput before inputs name, evalOutput after inputs name with
  | .ok lhs, .ok rhs => lhs == rhs
  | _, _ => false

def canEliminateIdentity (graph : FragmentGraph) : Bool :=
  match graph.nodes with
  | [idNode, next] =>
      idNode.op = .identity &&
      idNode.outputs.length = 1 &&
      next.inputs.any (· = idNode.outputs.head!)
  | _ => false

def eliminateIdentity (graph : FragmentGraph) : FragmentGraph :=
  match graph.nodes with
  | [idNode, next] =>
      let replacement := idNode.inputs.headD ""
      let oldName := idNode.outputs.headD ""
      let next' := { next with inputs := next.inputs.map fun input => if input = oldName then replacement else input }
      { graph with nodes := [next'] }
  | _ => graph

def canEliminateUnsqueezeSqueeze (graph : FragmentGraph) : Bool :=
  match graph.nodes with
  | [uNode, sNode] =>
      uNode.op = .unsqueeze && sNode.op = .squeeze &&
      uNode.outputs = sNode.inputs &&
      uNode.attrNats? "axes" = sNode.attrNats? "axes"
  | _ => false

def eliminateUnsqueezeSqueeze (graph : FragmentGraph) : FragmentGraph :=
  match graph.nodes with
  | [uNode, sNode] =>
      { graph with
        nodes := [{ op := .identity, inputs := uNode.inputs, outputs := sNode.outputs }] }
  | _ => graph

def canEliminateFullSlice (graph : FragmentGraph) : Bool :=
  match graph.nodes with
  | [sliceNode, idNode] =>
      sliceNode.op = .slice && idNode.op = .identity &&
      sliceNode.outputs = idNode.inputs &&
      (sliceNode.attrNats? "starts" |>.getD []).all (· = 0) &&
      attrBool sliceNode "full_slice_cert"
  | _ => false

def eliminateFullSlice (graph : FragmentGraph) : FragmentGraph :=
  match graph.nodes with
  | [sliceNode, idNode] =>
      { graph with nodes := [{ op := .identity, inputs := sliceNode.inputs, outputs := idNode.outputs }] }
  | _ => graph

def canFuseReshapeChain (graph : FragmentGraph) : Bool :=
  match graph.nodes with
  | [r1, r2] =>
      r1.op = .reshape && r2.op = .reshape && r1.outputs = r2.inputs &&
      r1.inputs.length = 1 && r2.outputs.length = 1
  | _ => false

def fuseReshapeChain (graph : FragmentGraph) : FragmentGraph :=
  match graph.nodes with
  | [r1, r2] =>
      { graph with
        nodes := [{ r2 with inputs := r1.inputs }] }
  | _ => graph

def coreEnv : Env :=
  [ ("x", .ints [2, 2] [1, 2, 3, 4])
  , ("bias", .ints [2, 2] [10, 20, 30, 40]) ]

def identityBefore : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .identity, inputs := ["x"], outputs := ["tmp"] }
      , { op := .add, inputs := ["tmp", "bias"], outputs := ["out"] } ]
    outputs := [] }

def identityReject : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .identity, inputs := ["x"], outputs := ["tmp"] }
      , { op := .mul, inputs := ["bias", "x"], outputs := ["out"] } ]
    outputs := [] }

def unsqueezeBefore : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .unsqueeze, inputs := ["x"], outputs := ["tmp"], attrs := [("axes", .nats [1])] }
      , { op := .squeeze, inputs := ["tmp"], outputs := ["out"], attrs := [("axes", .nats [1])] } ]
    outputs := [] }

def unsqueezeReject : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .unsqueeze, inputs := ["x"], outputs := ["tmp"], attrs := [("axes", .nats [0])] }
      , { op := .squeeze, inputs := ["tmp"], outputs := ["out"], attrs := [("axes", .nats [1])] } ]
    outputs := [] }

def sliceBefore : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .slice, inputs := ["x"], outputs := ["tmp"]
          attrs :=
            [ ("starts", .nats [0, 0])
            , ("ends", .nats [2, 2])
            , ("axes", .nats [0, 1])
            , ("full_slice_cert", .bool true) ] }
      , { op := .identity, inputs := ["tmp"], outputs := ["out"] } ]
    outputs := [] }

def sliceReject : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .slice, inputs := ["x"], outputs := ["tmp"]
          attrs := [("starts", .nats [0, 1]), ("ends", .nats [2, 2]), ("axes", .nats [0, 1])] }
      , { op := .identity, inputs := ["tmp"], outputs := ["out"] } ]
    outputs := [] }

def sliceMissingCert : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .slice, inputs := ["x"], outputs := ["tmp"]
          attrs := [("starts", .nats [0, 0]), ("ends", .nats [2, 2]), ("axes", .nats [0, 1])] }
      , { op := .identity, inputs := ["tmp"], outputs := ["out"] } ]
    outputs := [] }

def reshapeBefore : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .reshape, inputs := ["x"], outputs := ["tmp"], attrs := [("shape", .shape (.ofKnown [1, 4]))] }
      , { op := .reshape, inputs := ["tmp"], outputs := ["out"], attrs := [("shape", .shape (.ofKnown [2, 2]))] } ]
    outputs := [] }

def reshapeReject : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .reshape, inputs := ["x"], outputs := ["tmp"], attrs := [("shape", .shape (.ofKnown [1, 4]))] }
      , { op := .identity, inputs := ["tmp"], outputs := ["out"] } ]
    outputs := [] }

theorem identity_elimination_semantic_eq :
    outputsAgree identityBefore (eliminateIdentity identityBefore) coreEnv = true := by native_decide

theorem unsqueeze_elimination_semantic_eq :
    outputsAgree unsqueezeBefore (eliminateUnsqueezeSqueeze unsqueezeBefore) coreEnv = true := by
  native_decide

theorem slice_elimination_semantic_eq :
    outputsAgree sliceBefore (eliminateFullSlice sliceBefore) coreEnv = true := by native_decide

theorem reshape_fusion_semantic_eq :
    outputsAgree reshapeBefore (fuseReshapeChain reshapeBefore) coreEnv = true := by native_decide

end HeytingLean.Bridge.Abbott.ONNX.RewriteCore
