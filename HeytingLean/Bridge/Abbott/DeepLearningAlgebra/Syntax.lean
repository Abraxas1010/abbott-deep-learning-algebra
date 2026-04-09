namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra

/-- UID-addressable slot classes used by the paper's placeholder/configuration layer. -/
inductive SlotKind where
  | axis
  | dtype
  | opParam
  | placeholder
  deriving Repr, DecidableEq, Inhabited, BEq

instance : Hashable SlotKind where
  hash
    | .axis => hash (0 : Nat)
    | .dtype => hash (1 : Nat)
    | .opParam => hash (2 : Nat)
    | .placeholder => hash (3 : Nat)

/-- Reference to an open slot in an object or morphism term. -/
structure SlotRef where
  uid : Nat
  kind : SlotKind
  label : String := ""
  deriving Repr, DecidableEq, Inhabited, BEq

instance : Hashable SlotRef where
  hash s := mixHash (hash s.uid) (mixHash (hash s.kind) (hash s.label))

inductive EntityKind where
  | object
  | morphism
  deriving Repr, DecidableEq, Inhabited

/-- Lightweight signature node for atoms in the constructed-term system. -/
structure Signature where
  name : String
  kind : EntityKind := .object
  slots : Array SlotRef := #[]
  deriving Repr, DecidableEq, Inhabited

inductive BaseType where
  | bool
  | int
  | float
  deriving Repr, DecidableEq, Inhabited

/-- Object side of the constructed-term system. -/
inductive ObjectTerm where
  | atom (sig : Signature)
  | tensor (dtype : BaseType) (axes : Array SlotRef)
  | product (factors : Array ObjectTerm)
  | placeholder (slot : SlotRef)
  deriving Repr

/-- Morphism side of the constructed-term system. -/
inductive MorphismTerm where
  | id (obj : ObjectTerm)
  | primitive (name : String) (dom cod : ObjectTerm) (slots : Array SlotRef := #[])
  | compose (g f : MorphismTerm)
  | product (factors : Array MorphismTerm)
  | placeholder (slot : SlotRef) (dom cod : ObjectTerm)
  deriving Repr

/-- A root term pairs a body with its advertised interface and free input slots. -/
structure RootTerm where
  body : MorphismTerm
  output : ObjectTerm
  inputs : Array SlotRef := #[]
  deriving Repr

namespace ObjectTerm

mutual

partial def freeSlots : ObjectTerm → List SlotRef
  | .atom sig => sig.slots.toList
  | .tensor _ axes => axes.toList
  | .product factors => freeSlotsList factors.toList
  | .placeholder slot => [slot]

partial def freeSlotsList : List ObjectTerm → List SlotRef
  | [] => []
  | obj :: rest => obj.freeSlots ++ freeSlotsList rest

end

def fillSlot (uid : Nat) (replacement : ObjectTerm) : ObjectTerm → ObjectTerm
  | .atom sig => .atom sig
  | .tensor dtype axes => .tensor dtype axes
  | .product factors => .product (factors.map (fillSlot uid replacement))
  | .placeholder slot =>
      if slot.uid = uid then replacement else .placeholder slot

end ObjectTerm

namespace MorphismTerm

mutual

partial def freeSlots : MorphismTerm → List SlotRef
  | .id obj => obj.freeSlots
  | .primitive _ dom cod slots => dom.freeSlots ++ cod.freeSlots ++ slots.toList
  | .compose g f => g.freeSlots ++ f.freeSlots
  | .product factors => freeSlotsList factors.toList
  | .placeholder slot dom cod => slot :: (dom.freeSlots ++ cod.freeSlots)

partial def freeSlotsList : List MorphismTerm → List SlotRef
  | [] => []
  | hom :: rest => hom.freeSlots ++ freeSlotsList rest

end

def fillObjectSlot (uid : Nat) (replacement : ObjectTerm) : MorphismTerm → MorphismTerm
  | .id obj => .id (obj.fillSlot uid replacement)
  | .primitive name dom cod slots =>
      .primitive name (dom.fillSlot uid replacement) (cod.fillSlot uid replacement) slots
  | .compose g f =>
      .compose (fillObjectSlot uid replacement g) (fillObjectSlot uid replacement f)
  | .product factors =>
      .product (factors.map (fillObjectSlot uid replacement))
  | .placeholder slot dom cod =>
      .placeholder slot (dom.fillSlot uid replacement) (cod.fillSlot uid replacement)

def fillMorphismSlot (uid : Nat) (replacement : MorphismTerm) : MorphismTerm → MorphismTerm
  | .id obj => .id obj
  | .primitive name dom cod slots => .primitive name dom cod slots
  | .compose g f =>
      .compose (fillMorphismSlot uid replacement g) (fillMorphismSlot uid replacement f)
  | .product factors =>
      .product (factors.map (fillMorphismSlot uid replacement))
  | .placeholder slot dom cod =>
      if slot.uid = uid then replacement else .placeholder slot dom cod

end MorphismTerm

namespace RootTerm

def freeSlots (root : RootTerm) : List SlotRef :=
  root.inputs.toList ++ root.body.freeSlots ++ root.output.freeSlots

end RootTerm

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra
