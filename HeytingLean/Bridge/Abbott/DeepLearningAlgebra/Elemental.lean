import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.ProductCategory

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra

/-- Paper-adjacent elemental surface: named objects plus primitive morphisms. -/
structure ElementalCategory where
  name : String
  objects : Array ObjectTerm := #[]
  primitiveMorphisms : Array MorphismTerm := #[]
  deriving Repr

/-- Deterministic morphisms expose copy/discard preservation as explicit fields. -/
structure DeterministicMorphism where
  hom : MorphismTerm
  preservesCopy : Prop := True
  preservesDiscard : Prop := True
  deriving Repr

namespace DeterministicMorphism

def identity (obj : ObjectTerm) : DeterministicMorphism :=
  { hom := .id obj }

theorem identity_preservesCopy (obj : ObjectTerm) :
    (identity obj).preservesCopy := by
  trivial

theorem identity_preservesDiscard (obj : ObjectTerm) :
    (identity obj).preservesDiscard := by
  trivial

end DeterministicMorphism

/-- Product templates package deterministic blocks with named IO boundaries. -/
structure ProductTemplate where
  name : String
  dom : ProductObject
  cod : ProductObject
  core : DeterministicMorphism
  inputNames : Array String := #[]
  outputNames : Array String := #[]
  deriving Repr

namespace ProductTemplate

def asProductMorphism (template : ProductTemplate) : ProductMorphism :=
  { dom := template.dom
    cod := template.cod
    blocks := #[template.core.hom]
    blockInputs := #[]
    blockOutputs := #[]
    explicitWiring := #[] }

end ProductTemplate

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra
