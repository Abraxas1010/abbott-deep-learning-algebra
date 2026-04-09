import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples.Common
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples.Convolution

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples

open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples

def convolutionBaseInput : BroadcastObject :=
  BroadcastObject.singleton (eraseArrayLabels signalObject)

def convolutionBaseOutput : BroadcastObject :=
  BroadcastObject.singleton (eraseArrayLabels signalObject)

theorem convolutionBaseInput_toProduct :
    convolutionBaseInput.toProductObject = convTemplate.dom := by
  rfl

theorem convolutionBaseOutput_toProduct :
    convolutionBaseOutput.toProductObject = convTemplate.cod := by
  rfl

def convolutionBaseProductHom :
    ProductHom convolutionBaseInput.toProductObject convolutionBaseOutput.toProductObject :=
  castProductHom (templateProductHom convTemplate)
    convolutionBaseInput_toProduct.symm
    convolutionBaseOutput_toProduct.symm

def convolutionBaseHom : BroadcastHom convolutionBaseInput convolutionBaseOutput :=
  BroadcastHom.ofBase convolutionBaseProductHom

def convolutionWindowing : BroadcastHom
    (BroadcastObject.singleton signalObject)
    (BroadcastObject.singleton {
      dtype := .float
      axes := { axes := #[batchAxis, channelAxis, lengthAxis, kernelWidthAxis] } }) :=
  BroadcastHom.ofReindexing convolution.reindexings[0]

theorem convolution_base_refines_finite_template :
    convolutionBaseProductHom.toMorphism = convTemplate.asProductMorphism := rfl

theorem convolution_refines_finite_example :
    convolutionBaseProductHom.toMorphism = convTemplate.asProductMorphism ∧
      convolutionWindowing = BroadcastHom.ofReindexing convolution.reindexings[0] := by
  constructor <;> rfl

theorem convolution_erased_shape :
    (eraseArrayLabels signalObject).shape = signalObject.shape := by
  rfl

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples
