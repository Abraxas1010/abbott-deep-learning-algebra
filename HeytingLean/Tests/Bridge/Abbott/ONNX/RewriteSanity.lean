import HeytingLean.Bridge.Abbott.ONNX.Rewrite.Fusion

namespace HeytingLean.Tests.Bridge.Abbott.ONNX.RewriteSanity

open HeytingLean.Bridge.Abbott.ONNX.RewriteCore
open HeytingLean.Bridge.Abbott.ONNX.RewriteFusion

example : canEliminateIdentity identityBefore = true := rfl
example : canEliminateIdentity identityReject = false := rfl

example : canEliminateUnsqueezeSqueeze unsqueezeBefore = true := rfl
example : canEliminateUnsqueezeSqueeze unsqueezeReject = false := rfl

example : canEliminateFullSlice sliceBefore = true := rfl
example : canEliminateFullSlice sliceReject = false := rfl
example : canEliminateFullSlice sliceMissingCert = false := rfl

example : canFuseReshapeChain reshapeBefore = true := rfl
example : canFuseReshapeChain reshapeReject = false := rfl

example : canFuseConvAdd convAddBefore = true := rfl
example : canFuseConvAdd convAddReject = false := rfl
example : canFuseConvAdd convAddMissingCert = false := rfl

example : canFuseConvMul convMulBefore = true := rfl
example : canFuseConvMul convMulReject = false := rfl
example : canFuseConvMul convMulMissingWitness = false := rfl

example : canFuseConvBatchNorm convBatchNormBefore = true := rfl
example : canFuseConvBatchNorm convBatchNormReject = false := rfl
example : canFuseConvBatchNorm convBatchNormMissingWitness = false := rfl

example : canFuseReluClip reluClipBefore = true := rfl
example : canFuseReluClip reluClipReject = false := rfl
example : canFuseReluClip reluClipMaxReject = false := rfl

example : canFuseMatMulAdd matmulAddBefore = true := rfl
example : canFuseMatMulAdd matmulAddReject = false := rfl
example : canFuseMatMulAdd matmulAddMissingCert = false := rfl

example : canDropGemmActivation gemmActivationBefore = true := rfl
example : canDropGemmActivation gemmActivationReject = false := rfl
example : canDropGemmActivation gemmActivationMissingCert = false := rfl

example : canDropConvActivation convActivationBefore = true := rfl
example : canDropConvActivation convActivationReject = false := rfl
example : canDropConvActivation convActivationMissingCert = false := rfl

end HeytingLean.Tests.Bridge.Abbott.ONNX.RewriteSanity
