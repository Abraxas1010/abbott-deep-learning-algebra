import HeytingLean.Bridge.Abbott.ONNX.Coverage

namespace HeytingLean.Tests.Bridge.Abbott.ONNX.CoverageSanity

open HeytingLean.Bridge.Abbott.ONNX

example : supportedOp .reshape = true := rfl

example : supportStatus .conv = .supported := rfl

example : supportStatus .resize = .unsupported := rfl

example : countStatus .supported = 24 := rfl

example : countStatus .partialSupport = 5 := rfl

example : countStatus .unsupported = 1 := rfl

example : coverageLedger.length = 30 := coverageLedger_length

example : coverageLedger.any (fun entry => decide (entry.snd = .supported)) = true :=
  supportedOp_not_all_false

example : coverageLedger.any (fun entry => decide (entry.snd = .partialSupport)) = true := by
  simpa using supportStatus_has_partial

example : coverageLedger.any (fun entry => decide (entry.snd = .unsupported)) = true :=
  supportStatus_has_unsupported

end HeytingLean.Tests.Bridge.Abbott.ONNX.CoverageSanity
