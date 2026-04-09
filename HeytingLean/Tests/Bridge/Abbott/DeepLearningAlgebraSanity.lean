import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Syntax
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.ProductCategory
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Elemental

namespace HeytingLean.Tests.Bridge.Abbott

open HeytingLean.Bridge.Abbott.DeepLearningAlgebra

def axis0 : SlotRef := { uid := 0, kind := .axis, label := "batch" }
def axis1 : SlotRef := { uid := 1, kind := .axis, label := "channel" }
def kernelSlot : SlotRef := { uid := 10, kind := .opParam, label := "kernel" }

def sampleTensor : ObjectTerm :=
  .tensor .float #[axis0, axis1]

def samplePlaceholder : ObjectTerm :=
  .placeholder axis0

def samplePrimitive : MorphismTerm :=
  .primitive "relu" sampleTensor sampleTensor #[kernelSlot]

def sampleTemplate : ProductTemplate :=
  { name := "relu-template"
    dom := ProductObject.singleton sampleTensor
    cod := ProductObject.singleton sampleTensor
    core := { hom := samplePrimitive } }

#check SlotRef
#check ObjectTerm
#check MorphismTerm
#check ProductObject
#check ProductMorphism
#check ProductTemplate

example : axis0.kind = .axis := rfl

example : sampleTensor = .tensor .float #[axis0, axis1] := rfl

example : samplePlaceholder = .placeholder axis0 := rfl

example : samplePrimitive = .primitive "relu" sampleTensor sampleTensor #[kernelSlot] := rfl

example : (ProductObject.append (ProductObject.singleton sampleTensor)
    (ProductObject.singleton samplePlaceholder)).factorCount = 2 := rfl

example : (ProductMorphism.id (ProductObject.singleton sampleTensor)).blockCount = 0 := rfl

example : sampleTemplate.asProductMorphism.blockCount = 1 := rfl

end HeytingLean.Tests.Bridge.Abbott
