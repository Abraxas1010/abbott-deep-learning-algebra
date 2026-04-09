import Lake
open Lake DSL

package «abbott-deep-learning-algebra» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.24.0"

@[default_target]
lean_lib HeytingLean
