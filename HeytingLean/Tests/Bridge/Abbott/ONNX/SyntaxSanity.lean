import HeytingLean.Bridge.Abbott.ONNX.Syntax

namespace HeytingLean.Tests.Bridge.Abbott.ONNX.SyntaxSanity

open HeytingLean.Bridge.Abbott.ONNX

def imageInput : ValueDecl :=
  { name := "input"
    ty := { dtype := .float32, shape := .ofKnown [1, 3, 224, 224] } }

def weightInit : ValueDecl :=
  { name := "weight"
    ty := { dtype := .float32, shape := .ofKnown [16, 3, 3, 3] } }

def convNode : Node :=
  { op := .conv
    inputs := ["input", "weight"]
    outputs := ["conv_out"]
    attrs := [("strides", .nats [2, 2]), ("pads", .nats [1, 1, 1, 1])] }

def reluNode : Node :=
  { op := .relu
    inputs := ["conv_out"]
    outputs := ["relu_out"] }

def sampleGraph : FragmentGraph :=
  { inputs := [imageInput]
    initializers := [weightInit]
    nodes := [convNode, reluNode]
    outputs :=
      [{ name := "relu_out"
         ty := { dtype := .float32, shape := .ofKnown [1, 16, 112, 112] } }] }

example : RankedOp.all.length = 30 := rankedOp_all_length

example : ShapeExpr.rank imageInput.ty.shape = 4 := rfl

example : imageInput.ty.shape.knownDims? = some [1, 3, 224, 224] := rfl

example : AttrMap.find? convNode.attrs "strides" = some (.nats [2, 2]) := rfl

example : sampleGraph.nodes.length = 2 := rfl

example : sampleGraph.outputs.head?.map (·.name) = some "relu_out" := rfl

example : sampleGraph.initializers.length = 1 := rfl

end HeytingLean.Tests.Bridge.Abbott.ONNX.SyntaxSanity
