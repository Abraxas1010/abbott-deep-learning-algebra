import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples.Common
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples.Attention

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples

open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples

def attentionBaseInput : BroadcastObject :=
  BroadcastObject.singleton (eraseArrayLabels attentionObject)

def attentionBaseOutput : BroadcastObject :=
  BroadcastObject.singleton (eraseArrayLabels attentionObject)

theorem attentionBaseInput_toProduct :
    attentionBaseInput.toProductObject = attentionTemplate.dom := by
  rfl

theorem attentionBaseOutput_toProduct :
    attentionBaseOutput.toProductObject = attentionTemplate.cod := by
  rfl

def attentionBaseProductHom :
    ProductHom attentionBaseInput.toProductObject attentionBaseOutput.toProductObject :=
  castProductHom (templateProductHom attentionTemplate)
    attentionBaseInput_toProduct.symm
    attentionBaseOutput_toProduct.symm

def attentionBaseHom : BroadcastHom attentionBaseInput attentionBaseOutput :=
  BroadcastHom.ofBase attentionBaseProductHom

def attentionInputWeave : BroadcastHom
    (BroadcastObject.singleton attentionObject)
    (BroadcastObject.singleton attentionObject) :=
  BroadcastHom.ofWeave .float attention.inputWeaves[0] rfl

theorem attention_base_refines_finite_template :
    attentionBaseProductHom.toMorphism = attentionTemplate.asProductMorphism := rfl

theorem attention_refines_finite_example :
    attentionBaseProductHom.toMorphism = attentionTemplate.asProductMorphism ∧
      attentionInputWeave = BroadcastHom.ofWeave .float attention.inputWeaves[0] rfl := by
  constructor <;> rfl

theorem attention_erased_shape :
    (eraseArrayLabels attentionObject).shape = attentionObject.shape := by
  rfl

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples
