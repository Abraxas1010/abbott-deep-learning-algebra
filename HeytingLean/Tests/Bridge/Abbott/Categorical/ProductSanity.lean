import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical.Product

namespace HeytingLean.Tests.Bridge.Abbott.Categorical.ProductSanity

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

def objA : ObjectTerm := .atom { name := "A" }
def objB : ObjectTerm := .atom { name := "B" }
def objC : ObjectTerm := .atom { name := "C" }
def objD : ObjectTerm := .atom { name := "D" }

def prodAB : ProductObject := .singleton objA
def prodBC : ProductObject := .singleton objB
def prodCD : ProductObject := .singleton objC
def prodDE : ProductObject := .singleton objD

def wire0 : Wiring :=
  { source := { block := 0, factor := 0 }
    target := { block := 1, factor := 0 } }

def wire1 : Wiring :=
  { source := { block := 1, factor := 0 }
    target := { block := 2, factor := 0 } }

def wire2 : Wiring :=
  { source := { block := 2, factor := 0 }
    target := { block := 3, factor := 0 } }

def blockF : MorphismTerm := .primitive "f" objA objB
def blockG : MorphismTerm := .primitive "g" objB objC
def blockH : MorphismTerm := .primitive "h" objC objD

def homF : ProductHom prodAB prodBC where
  toMorphism :=
    { dom := prodAB
      cod := prodBC
      blocks := #[blockF]
      blockInputs := #[{ block := 0, factor := 0 }]
      blockOutputs := #[{ block := 1, factor := 0 }]
      explicitWiring := #[wire0] }
  dom_eq := rfl
  cod_eq := rfl

def homG : ProductHom prodBC prodCD where
  toMorphism :=
    { dom := prodBC
      cod := prodCD
      blocks := #[blockG]
      blockInputs := #[{ block := 1, factor := 0 }]
      blockOutputs := #[{ block := 2, factor := 0 }]
      explicitWiring := #[wire1] }
  dom_eq := rfl
  cod_eq := rfl

def homH : ProductHom prodCD prodDE where
  toMorphism :=
    { dom := prodCD
      cod := prodDE
      blocks := #[blockH]
      blockInputs := #[{ block := 2, factor := 0 }]
      blockOutputs := #[{ block := 3, factor := 0 }]
      explicitWiring := #[wire2] }
  dom_eq := rfl
  cod_eq := rfl

example :
    (ProductHom.comp homF homG).toMorphism.explicitWiring ≠
      (ProductHom.comp homG homH).toMorphism.explicitWiring := by
  native_decide

example :
    ProductHom.comp (ProductHom.comp homF homG) homH =
      ProductHom.comp homF (ProductHom.comp homG homH) := by
  exact ProductHom.ext_toMorphism _ _ (ProductHom.assoc_toMorphism homF homG homH)

example :
    (ProductHom.comp (ProductHom.comp homF homG) homH).toMorphism.blocks =
      (ProductHom.comp homF (ProductHom.comp homG homH)).toMorphism.blocks := by
  rfl

end HeytingLean.Tests.Bridge.Abbott.Categorical.ProductSanity
