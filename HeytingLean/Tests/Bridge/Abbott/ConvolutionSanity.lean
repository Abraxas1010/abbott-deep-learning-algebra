import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples.Convolution
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples.Attention

namespace HeytingLean.Tests.Bridge.Abbott

open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples

example : convolution.inputs.size = 2 := rfl
example : convolution.explicitBroadcasting = true := rfl
example : attention.inputs.size = 3 := rfl
example : attention.outputWeaves.size = 1 := rfl

end HeytingLean.Tests.Bridge.Abbott
