import Lean
import HeytingLean.Bridge.Abbott.ONNX.Syntax

namespace HeytingLean.Bridge.Abbott.ONNX

/-- Honest support status for the canonical Abbott ONNX fragment. -/
inductive SupportStatus where
  | supported
  | partialSupport
  | unsupported
deriving Repr, Inhabited, DecidableEq, BEq, Lean.ToJson, Lean.FromJson

def supportStatus : RankedOp → SupportStatus
  | .reshape => .supported
  | .transpose => .supported
  | .expand => .supported
  | .unsqueeze => .supported
  | .squeeze => .supported
  | .add => .supported
  | .mul => .supported
  | .matMul => .supported
  | .gemm => .supported
  | .conv => .supported
  | .attention => .partialSupport
  | .flatten => .supported
  | .concat => .supported
  | .slice => .supported
  | .gather => .supported
  | .shape => .supported
  | .constantOfShape => .supported
  | .batchNormalization => .partialSupport
  | .relu => .supported
  | .clip => .partialSupport
  | .softmax => .supported
  | .reduceSum => .supported
  | .reduceMean => .partialSupport
  | .maxPool => .supported
  | .averagePool => .supported
  | .where => .supported
  | .identity => .supported
  | .cast => .supported
  | .pad => .partialSupport
  | .resize => .unsupported

def supportedOp (op : RankedOp) : Bool :=
  decide (supportStatus op = .supported)

def coverageLedger : List (RankedOp × SupportStatus) :=
  RankedOp.all.map fun op => (op, supportStatus op)

def countStatus (status : SupportStatus) : Nat :=
  coverageLedger.countP fun entry => decide (entry.snd = status)

theorem supportedOp_not_all_false :
    coverageLedger.any (fun entry => decide (entry.snd = .supported)) = true := by
  native_decide

theorem supportStatus_has_partial :
    coverageLedger.any (fun entry => decide (entry.snd = .partialSupport)) = true := by
  native_decide

theorem supportStatus_has_unsupported :
    coverageLedger.any (fun entry => decide (entry.snd = .unsupported)) = true := by
  native_decide

theorem coverageLedger_length : coverageLedger.length = 30 := by
  simp [coverageLedger, rankedOp_all_length]

end HeytingLean.Bridge.Abbott.ONNX
