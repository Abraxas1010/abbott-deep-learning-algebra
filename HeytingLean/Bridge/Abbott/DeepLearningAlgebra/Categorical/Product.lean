import Mathlib.CategoryTheory.Category.Basic
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.ProductCategory
import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Remapping

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical

open CategoryTheory
open HeytingLean.Bridge.Abbott.DeepLearningAlgebra

structure ProductHom (X Y : ProductObject) where
  toMorphism : ProductMorphism
  dom_eq : toMorphism.dom = X
  cod_eq : toMorphism.cod = Y
  deriving Repr

namespace ProductHom

@[simp] theorem dom (f : ProductHom X Y) : f.toMorphism.dom = X :=
  f.dom_eq

@[simp] theorem cod (f : ProductHom X Y) : f.toMorphism.cod = Y :=
  f.cod_eq

@[simp] def id (X : ProductObject) : ProductHom X X where
  toMorphism := ProductMorphism.id X
  dom_eq := rfl
  cod_eq := rfl

@[simp] def comp (f : ProductHom X Y) (g : ProductHom Y Z) : ProductHom X Z where
  toMorphism := ProductMorphism.sequential f.toMorphism g.toMorphism
  dom_eq := by simp [ProductMorphism.sequential]
  cod_eq := by simp [ProductMorphism.sequential]

@[ext] theorem ext_toMorphism (f g : ProductHom X Y)
    (h : f.toMorphism = g.toMorphism) : f = g := by
  cases f
  cases g
  cases h
  simp

@[ext] theorem ext (f g : ProductHom X Y)
    (hBlocks : f.toMorphism.blocks = g.toMorphism.blocks)
    (hInputs : f.toMorphism.blockInputs = g.toMorphism.blockInputs)
    (hOutputs : f.toMorphism.blockOutputs = g.toMorphism.blockOutputs)
    (hWiring : f.toMorphism.explicitWiring = g.toMorphism.explicitWiring) : f = g := by
  apply ext_toMorphism
  cases f with
  | mk fm fdom fcod =>
      cases g with
      | mk gm gdom gcod =>
          cases fm with
          | mk fDom fCod fBlocks fInputs fOutputs fWiring =>
              cases gm with
              | mk gDom gCod gBlocks gInputs gOutputs gWiring =>
                  simp at fdom fcod gdom gcod
                  subst fDom
                  subst fCod
                  subst gDom
                  subst gCod
                  cases hBlocks
                  cases hInputs
                  cases hOutputs
                  cases hWiring
                  rfl

@[simp] theorem id_comp_toMorphism (f : ProductHom X Y) :
    (comp (id X) f).toMorphism = f.toMorphism := by
  cases f with
  | mk fm fdom fcod =>
      cases fm
      cases fdom
      cases fcod
      simp [comp, id, ProductMorphism.sequential, ProductMorphism.id]

@[simp] theorem comp_id_toMorphism (f : ProductHom X Y) :
    (comp f (id Y)).toMorphism = f.toMorphism := by
  cases f with
  | mk fm fdom fcod =>
      cases fm
      cases fdom
      cases fcod
      simp [comp, id, ProductMorphism.sequential, ProductMorphism.id]

@[simp] theorem assoc_toMorphism (f : ProductHom W X) (g : ProductHom X Y) (h : ProductHom Y Z) :
    (comp (comp f g) h).toMorphism = (comp f (comp g h)).toMorphism := by
  cases f with
  | mk fm fdom fcod =>
      cases g with
      | mk gm gdom gcod =>
          cases h with
          | mk hm hdom hcod =>
              cases fm
              cases gm
              cases hm
              simp [comp, ProductMorphism.sequential, Array.append_assoc]

instance : Category ProductObject where
  Hom X Y := ProductHom X Y
  id := ProductHom.id
  comp f g := ProductHom.comp f g
  id_comp := by
    intro X Y f
    exact ext_toMorphism _ _ (id_comp_toMorphism f)
  comp_id := by
    intro X Y f
    exact ext_toMorphism _ _ (comp_id_toMorphism f)
  assoc := by
    intro W X Y Z f g h
    exact ext_toMorphism _ _ (assoc_toMorphism f g h)

end ProductHom

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Categorical
