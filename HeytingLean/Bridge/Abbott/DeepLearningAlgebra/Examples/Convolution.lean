import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Laws

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples

open HeytingLean.Bridge.Abbott.DeepLearningAlgebra

def batchAxis : Axis := { uid := 0, label := "batch", size := 8 }
def channelAxis : Axis := { uid := 1, label := "channel", size := 16 }
def lengthAxis : Axis := { uid := 2, label := "length", size := 64 }
def kernelWidthAxis : Axis := { uid := 3, label := "kernel_width", size := 3 }

def signalObject : ArrayObject :=
  { dtype := .float
    axes := { axes := #[batchAxis, channelAxis, lengthAxis] } }

def kernelObject : ArrayObject :=
  { dtype := .float
    axes := { axes := #[channelAxis, kernelWidthAxis] } }

def convTemplate : ProductTemplate :=
  { name := "conv1d"
    dom := { factors := #[.tensor .float #[{ uid := 0, kind := .axis }, { uid := 1, kind := .axis }, { uid := 2, kind := .axis }]] }
    cod := { factors := #[.tensor .float #[{ uid := 0, kind := .axis }, { uid := 1, kind := .axis }, { uid := 2, kind := .axis }]] }
    core := { hom := .primitive "conv1d" (.tensor .float #[]) (.tensor .float #[]) #[{ uid := 10, kind := .opParam }, { uid := 11, kind := .opParam }] } }

def convolution : BroadcastedOperation :=
  { name := "convolution"
    inputs := #[signalObject, kernelObject]
    output := signalObject
    base := convTemplate
    reindexings := #[{
      dom := signalObject
      cod := { dtype := .float, axes := { axes := #[batchAxis, channelAxis, lengthAxis, kernelWidthAxis] } }
      stride := {
        dom := signalObject.axes
        cod := { axes := #[batchAxis, channelAxis, lengthAxis, kernelWidthAxis] }
        linear := #[
          #[1, 0, 0, 0],
          #[0, 1, 0, 0],
          #[0, 0, 1, 1]
        ]
        offset := #[0, 0, -1]
      }
    }]
    inputWeaves := #[{
      source := { axes := #[batchAxis, channelAxis, lengthAxis, kernelWidthAxis] }
      target := { axes := #[batchAxis, channelAxis, lengthAxis, kernelWidthAxis] }
      slotOrder := #[0, 1, 2, 3]
    }]
    outputWeaves := #[Weave.identity signalObject.axes] }

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples
