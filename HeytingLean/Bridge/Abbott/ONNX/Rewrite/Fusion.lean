import HeytingLean.Bridge.Abbott.ONNX.Rewrite.Core

namespace HeytingLean.Bridge.Abbott.ONNX.RewriteFusion

open HeytingLean.Bridge.Abbott.ONNX
open HeytingLean.Bridge.Abbott.ONNX.RewriteCore

private def attrBool (node : Node) (key : String) : Bool :=
  match AttrMap.find? node.attrs key with
  | some (.bool value) => value
  | some (.int value) => value ≠ 0
  | some (.nat value) => value ≠ 0
  | _ => false

private def attrStr? (node : Node) (key : String) : Option String :=
  match AttrMap.find? node.attrs key with
  | some (.str value) => some value
  | _ => none

private def certifiedInputs? (node : Node) (arity : Nat) : Option (List String) :=
  (List.range arity).mapM fun idx => attrStr? node s!"fused_input_{idx}"

def convEnv : Env :=
  [ ("x", .ints [1, 1, 3, 3] [1, 2, 3, 4, 5, 6, 7, 8, 9])
  , ("xf", .floats [1, 1, 3, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
  , ("kernel", .ints [1, 1, 2, 2] [1, 0, 0, 1])
  , ("kernel2", .ints [1, 1, 2, 2] [2, 0, 0, 2])
  , ("kernelf", .floats [1, 1, 2, 2] [1.0, 0.0, 0.0, 1.0])
  , ("kernelf2", .floats [1, 1, 2, 2] [2.0, 0.0, 0.0, 2.0])
  , ("biasMap", .ints [1, 1, 1, 1] [1])
  , ("biasWide", .ints [1, 1, 2, 2] [1, 2, 3, 4])
  , ("bias1", .ints [1] [1])
  , ("scaleMap", .ints [1, 1, 1, 1] [2])
  , ("scale", .floats [1] [2.0])
  , ("fbias", .floats [1] [0.0])
  , ("mean", .floats [1] [0.0])
  , ("var", .floats [1] [1.0]) ]

def linearEnv : Env :=
  [ ("a", .ints [2, 3] [1, 2, 3, 4, 5, 6])
  , ("b", .ints [3, 2] [7, 8, 9, 10, 11, 12])
  , ("bias", .ints [2] [1, 2])
  , ("fa", .ints [2, 2] [1, 1, 1, 1])
  , ("fb", .ints [2, 2] [1, 2, 3, 4])
  , ("bigBias", .ints [2] [10, 10]) ]

def canFuseConvAdd (graph : FragmentGraph) : Bool :=
  match graph.nodes with
  | [convNode, addNode] =>
      convNode.op = .conv && addNode.op = .add &&
      convNode.outputs = [addNode.inputs.headD ""] && addNode.inputs.length = 2 &&
      attrBool addNode "bias_broadcast_cert"
  | _ => false

def fuseConvAdd (graph : FragmentGraph) : FragmentGraph :=
  match graph.nodes with
  | [convNode, addNode] =>
      { graph with
        nodes := [{ convNode with inputs := convNode.inputs ++ [addNode.inputs.getD 1 ""] , outputs := addNode.outputs }] }
  | _ => graph

def canFuseConvMul (graph : FragmentGraph) : Bool :=
  match graph.nodes with
  | [convNode, mulNode] =>
      convNode.op = .conv && mulNode.op = .mul &&
      convNode.outputs = [mulNode.inputs.headD ""] &&
      (certifiedInputs? mulNode 2).isSome &&
      attrBool mulNode "scale_channel_cert"
  | _ => false

def fuseConvMul (graph : FragmentGraph) : FragmentGraph :=
  match graph.nodes with
  | [convNode, mulNode] =>
      match certifiedInputs? mulNode 2 with
      | some inputs =>
          { graph with nodes := [{ convNode with inputs := inputs, outputs := mulNode.outputs }] }
      | none => graph
  | _ => graph

def canFuseConvBatchNorm (graph : FragmentGraph) : Bool :=
  match graph.nodes with
  | [convNode, bnNode] =>
      convNode.op = .conv && bnNode.op = .batchNormalization &&
      convNode.outputs = [bnNode.inputs.headD ""] &&
      bnNode.inputs.length = 5 &&
      (certifiedInputs? bnNode 2).isSome &&
      attrBool bnNode "batchnorm_fold_cert"
  | _ => false

def fuseConvBatchNorm (graph : FragmentGraph) : FragmentGraph :=
  match graph.nodes with
  | [convNode, bnNode] =>
      match certifiedInputs? bnNode 2 with
      | some inputs =>
          { graph with nodes := [{ convNode with inputs := inputs, outputs := bnNode.outputs }] }
      | none => graph
  | _ => graph

def canFuseReluClip (graph : FragmentGraph) : Bool :=
  match graph.nodes with
  | [reluNode, clipNode] =>
      reluNode.op = .relu && clipNode.op = .clip &&
      reluNode.outputs = clipNode.inputs &&
      clipNode.attrInt? "min" = some 0 &&
      clipNode.attrInt? "max" = none
  | _ => false

def fuseReluClip (graph : FragmentGraph) : FragmentGraph :=
  match graph.nodes with
  | [reluNode, clipNode] =>
      { graph with nodes := [{ reluNode with outputs := clipNode.outputs }] }
  | _ => graph

def canFuseMatMulAdd (graph : FragmentGraph) : Bool :=
  match graph.nodes with
  | [mmNode, addNode] =>
      mmNode.op = .matMul && addNode.op = .add &&
      mmNode.outputs = [addNode.inputs.headD ""] &&
      attrBool addNode "gemm_bias_cert"
  | _ => false

def fuseMatMulAdd (graph : FragmentGraph) : FragmentGraph :=
  match graph.nodes with
  | [mmNode, addNode] =>
      { graph with
        nodes := [{ op := .gemm, inputs := mmNode.inputs ++ [addNode.inputs.getD 1 ""], outputs := addNode.outputs }] }
  | _ => graph

def canDropGemmActivation (graph : FragmentGraph) : Bool :=
  match graph.nodes with
  | [gemmNode, reluNode] =>
      gemmNode.op = .gemm && reluNode.op = .relu &&
      gemmNode.outputs = reluNode.inputs &&
      attrBool reluNode "nonnegative_cert"
  | _ => false

def dropGemmActivation (graph : FragmentGraph) : FragmentGraph :=
  match graph.nodes with
  | [gemmNode, reluNode] => { graph with nodes := [{ gemmNode with outputs := reluNode.outputs }] }
  | _ => graph

def canDropConvActivation (graph : FragmentGraph) : Bool :=
  match graph.nodes with
  | [convNode, reluNode] =>
      convNode.op = .conv && reluNode.op = .relu &&
      convNode.outputs = reluNode.inputs &&
      attrBool reluNode "nonnegative_cert"
  | _ => false

def dropConvActivation (graph : FragmentGraph) : FragmentGraph :=
  match graph.nodes with
  | [convNode, reluNode] => { graph with nodes := [{ convNode with outputs := reluNode.outputs }] }
  | _ => graph

def convAddBefore : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .conv, inputs := ["x", "kernel"], outputs := ["tmp"] }
      , { op := .add, inputs := ["tmp", "biasMap"], outputs := ["out"]
          attrs := [("bias_broadcast_cert", .bool true)] } ]
    outputs := [] }

def convAddReject : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .conv, inputs := ["x", "kernel"], outputs := ["tmp"] }
      , { op := .mul, inputs := ["tmp", "biasWide"], outputs := ["out"] } ]
    outputs := [] }

def convAddMissingCert : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .conv, inputs := ["x", "kernel"], outputs := ["tmp"] }
      , { op := .add, inputs := ["tmp", "biasMap"], outputs := ["out"] } ]
    outputs := [] }

def convMulBefore : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .conv, inputs := ["x", "kernel"], outputs := ["tmp"] }
      , { op := .mul, inputs := ["tmp", "scaleMap"], outputs := ["out"]
          attrs :=
            [ ("scale_channel_cert", .bool true)
            , ("fused_input_0", .str "x")
            , ("fused_input_1", .str "kernel2") ] } ]
    outputs := [] }

def convMulReject : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .conv, inputs := ["x", "kernel"], outputs := ["tmp"] }
      , { op := .add, inputs := ["tmp", "scaleMap"], outputs := ["out"] } ]
    outputs := [] }

def convMulMissingWitness : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .conv, inputs := ["x", "kernel"], outputs := ["tmp"] }
      , { op := .mul, inputs := ["tmp", "scaleMap"], outputs := ["out"]
          attrs := [("scale_channel_cert", .bool true)] } ]
    outputs := [] }

def convBatchNormBefore : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .conv, inputs := ["xf", "kernelf"], outputs := ["tmp"] }
      , { op := .batchNormalization, inputs := ["tmp", "scale", "fbias", "mean", "var"], outputs := ["out"]
          attrs :=
            [ ("epsilon", .float 0.0)
            , ("batchnorm_fold_cert", .bool true)
            , ("fused_input_0", .str "xf")
            , ("fused_input_1", .str "kernelf2") ] } ]
    outputs := [] }

def convBatchNormReject : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .conv, inputs := ["xf", "kernelf"], outputs := ["tmp"] }
      , { op := .relu, inputs := ["tmp"], outputs := ["out"] } ]
    outputs := [] }

def convBatchNormMissingWitness : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .conv, inputs := ["xf", "kernelf"], outputs := ["tmp"] }
      , { op := .batchNormalization, inputs := ["tmp", "scale", "fbias", "mean", "var"], outputs := ["out"]
          attrs := [("epsilon", .float 0.0), ("batchnorm_fold_cert", .bool true)] } ]
    outputs := [] }

def reluClipBefore : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .relu, inputs := ["x"], outputs := ["tmp"] }
      , { op := .clip, inputs := ["tmp"], outputs := ["out"], attrs := [("min", .int 0)] } ]
    outputs := [] }

def reluClipReject : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .relu, inputs := ["x"], outputs := ["tmp"] }
      , { op := .clip, inputs := ["tmp"], outputs := ["out"], attrs := [("min", .int 1)] } ]
    outputs := [] }

def reluClipMaxReject : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .relu, inputs := ["x"], outputs := ["tmp"] }
      , { op := .clip, inputs := ["tmp"], outputs := ["out"], attrs := [("min", .int 0), ("max", .int 5)] } ]
    outputs := [] }

def matmulAddBefore : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .matMul, inputs := ["a", "b"], outputs := ["tmp"] }
      , { op := .add, inputs := ["tmp", "bias"], outputs := ["out"]
          attrs := [("gemm_bias_cert", .bool true)] } ]
    outputs := [] }

def matmulAddReject : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .matMul, inputs := ["a", "b"], outputs := ["tmp"] }
      , { op := .mul, inputs := ["tmp", "bias"], outputs := ["out"] } ]
    outputs := [] }

def matmulAddMissingCert : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .matMul, inputs := ["a", "b"], outputs := ["tmp"] }
      , { op := .add, inputs := ["tmp", "bias"], outputs := ["out"] } ]
    outputs := [] }

def gemmActivationBefore : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .gemm, inputs := ["fa", "fb", "bigBias"], outputs := ["tmp"] }
      , { op := .relu, inputs := ["tmp"], outputs := ["out"]
          attrs := [("nonnegative_cert", .bool true)] } ]
    outputs := [] }

def gemmActivationReject : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .gemm, inputs := ["fa", "fb", "bigBias"], outputs := ["tmp"] }
      , { op := .clip, inputs := ["tmp"], outputs := ["out"], attrs := [("min", .int 1)] } ]
    outputs := [] }

def gemmActivationMissingCert : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .gemm, inputs := ["fa", "fb", "bigBias"], outputs := ["tmp"] }
      , { op := .relu, inputs := ["tmp"], outputs := ["out"] } ]
    outputs := [] }

def convActivationBefore : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .conv, inputs := ["x", "kernel", "bias1"], outputs := ["tmp"] }
      , { op := .relu, inputs := ["tmp"], outputs := ["out"]
          attrs := [("nonnegative_cert", .bool true)] } ]
    outputs := [] }

def convActivationReject : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .conv, inputs := ["x", "kernel", "bias1"], outputs := ["tmp"] }
      , { op := .clip, inputs := ["tmp"], outputs := ["out"], attrs := [("min", .int 1)] } ]
    outputs := [] }

def convActivationMissingCert : FragmentGraph :=
  { inputs := []
    nodes :=
      [ { op := .conv, inputs := ["x", "kernel", "bias1"], outputs := ["tmp"] }
      , { op := .relu, inputs := ["tmp"], outputs := ["out"] } ]
    outputs := [] }

theorem conv_add_fusion_semantic_eq :
    outputsAgree convAddBefore (fuseConvAdd convAddBefore) convEnv = true := by native_decide

theorem conv_mul_fusion_semantic_eq :
    outputsAgree convMulBefore (fuseConvMul convMulBefore) convEnv = true := by native_decide

theorem conv_batchnorm_fusion_semantic_eq :
    outputsAgree convBatchNormBefore (fuseConvBatchNorm convBatchNormBefore) convEnv = true := by
  native_decide

theorem relu_clip_fusion_semantic_eq :
    outputsAgree reluClipBefore (fuseReluClip reluClipBefore) convEnv = true := by native_decide

theorem matmul_add_fusion_semantic_eq :
    outputsAgree matmulAddBefore (fuseMatMulAdd matmulAddBefore) linearEnv = true := by native_decide

theorem gemm_activation_semantic_eq :
    outputsAgree gemmActivationBefore (dropGemmActivation gemmActivationBefore) linearEnv = true := by
  native_decide

theorem conv_activation_semantic_eq :
    outputsAgree convActivationBefore (dropConvActivation convActivationBefore) convEnv = true := by
  native_decide

end HeytingLean.Bridge.Abbott.ONNX.RewriteFusion
