import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Laws

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples

open HeytingLean.Bridge.Abbott.DeepLearningAlgebra

def tokenAxis : Axis := { uid := 10, label := "token", size := 128 }
def featureAxis : Axis := { uid := 11, label := "feature", size := 64 }

def attentionObject : ArrayObject :=
  { dtype := .float
    axes := { axes := #[tokenAxis, featureAxis] } }

def attentionTemplate : ProductTemplate :=
  { name := "attention"
    dom := { factors := #[.tensor .float #[{ uid := 10, kind := .axis }, { uid := 11, kind := .axis }]] }
    cod := { factors := #[.tensor .float #[{ uid := 10, kind := .axis }, { uid := 11, kind := .axis }]] }
    core := { hom := .primitive "attention" (.tensor .float #[]) (.tensor .float #[]) } }

def attention : BroadcastedOperation :=
  { name := "self_attention"
    inputs := #[attentionObject, attentionObject, attentionObject]
    output := attentionObject
    base := attentionTemplate
    inputWeaves := #[Weave.identity attentionObject.axes]
    outputWeaves := #[Weave.identity attentionObject.axes] }

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra.Examples
