import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples.Convolution
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples.Attention

namespace HeytingLean.Tests.Bridge.Abbott.Categorical.ExamplesSanity

open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Examples

example :
    convolutionBaseProductHom.toMorphism =
      HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples.convTemplate.asProductMorphism := by
  exact convolution_base_refines_finite_template

example :
    attentionBaseProductHom.toMorphism =
      HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples.attentionTemplate.asProductMorphism := by
  exact attention_base_refines_finite_template

example :
    convolutionWindowing =
      HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.BroadcastHom.ofReindexing
        HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples.convolution.reindexings[0] := by
  exact (convolution_refines_finite_example).2

example :
    attentionInputWeave =
      HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.BroadcastHom.ofWeave .float
        HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples.attention.inputWeaves[0] rfl := by
  exact (attention_refines_finite_example).2

end HeytingLean.Tests.Bridge.Abbott.Categorical.ExamplesSanity
