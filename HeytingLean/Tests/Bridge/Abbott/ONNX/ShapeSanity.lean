import HeytingLean.Bridge.Abbott.ONNX.Shape

namespace HeytingLean.Tests.Bridge.Abbott.ONNX.ShapeSanity

open HeytingLean.Bridge.Abbott.ONNX

example : elementCount [2, 3, 4] = 24 := rfl

example : broadcastShape [3, 1, 5] [1, 4, 5] = .ok [3, 4, 5] := rfl

example : reshapeShape [2, 3, 4] [4, 6] = .ok [4, 6] := rfl

example : flattenShape [2, 3, 4] 1 = .ok [2, 12] := rfl

example : unsqueezeShape [2, 3] [1] = .ok [2, 1, 3] := rfl

example : squeezeShape [2, 1, 3] [1] = .ok [2, 3] := rfl

example : concatShape 1 [[2, 3], [2, 5], [2, 7]] = .ok [2, 15] := rfl

example : whereShape [1, 3] [2, 3] [2, 1] = .ok [2, 3] := rfl

end HeytingLean.Tests.Bridge.Abbott.ONNX.ShapeSanity
