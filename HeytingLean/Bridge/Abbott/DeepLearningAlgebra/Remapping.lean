import HeytingLean.Bridge.Abbott.DeepLearningAlgebra.ProductCategory

namespace HeytingLean.Bridge.Abbott.DeepLearningAlgebra

/-- Finite remappings model discrete index maps `Fin cod -> Fin dom`. -/
structure FiniteRemapping (dom cod : Nat) where
  toFun : Fin cod → Fin dom

namespace FiniteRemapping

def id (n : Nat) : FiniteRemapping n n :=
  { toFun := fun i => i }

def comp {a b c : Nat} (f : FiniteRemapping a b) (g : FiniteRemapping b c) :
    FiniteRemapping a c :=
  { toFun := fun i => f.toFun (g.toFun i) }

theorem comp_id {a b : Nat} (f : FiniteRemapping a b) :
    comp f (id b) = f := by
  cases f
  rfl

theorem id_comp {a b : Nat} (f : FiniteRemapping a b) :
    comp (id a) f = f := by
  cases f
  rfl

/-- Segment arity for a finite family of flat objects indexed by `Fin n`. -/
def segmentedArity : {n : Nat} -> (Fin n → Nat) -> Nat
  | 0, _ => 0
  | n + 1, sizes => sizes 0 + segmentedArity (fun i : Fin n => sizes i.succ)

@[simp] theorem segmentedArity_zero (sizes : Fin 0 → Nat) :
    segmentedArity sizes = 0 := rfl

@[simp] theorem segmentedArity_succ {n : Nat} (sizes : Fin (n + 1) → Nat) :
    segmentedArity sizes = sizes 0 + segmentedArity (fun i : Fin n => sizes i.succ) := rfl

/--
Binary direct sum of remappings. This is the paper's concatenation operation
for two discrete functions, expressed in the repo's `Fin cod -> Fin dom`
orientation.
-/
def directSum {a b c d : Nat} (μ : FiniteRemapping a b) (ν : FiniteRemapping c d) :
    FiniteRemapping (a + c) (b + d) where
  toFun := Fin.addCases
    (fun i => Fin.castAdd c (μ.toFun i))
    (fun i => Fin.natAdd a (ν.toFun i))

/-- Embed a segment-local offset into the flattened arity. -/
def joinSegments : {n : Nat} -> (sizes : Fin n → Nat) -> (seg : Fin n) ->
    Fin (sizes seg) -> Fin (segmentedArity sizes)
  | 0, _, seg, _ => nomatch seg
  | n + 1, sizes, seg, off =>
      Fin.cases
        (motive := fun seg => Fin (sizes seg) -> Fin (segmentedArity sizes))
        (fun off0 => by
          simpa [segmentedArity] using
            Fin.castAdd (segmentedArity (fun i : Fin n => sizes i.succ)) off0)
        (fun segTail offTail => by
          let recIdx : Fin (segmentedArity (fun i : Fin n => sizes i.succ)) :=
            joinSegments (fun i : Fin n => sizes i.succ) segTail (by simpa using offTail)
          simpa [segmentedArity] using Fin.natAdd (sizes 0) recIdx)
        seg off

/-- Recover segment choice and segment-local offset from a flattened index. -/
def splitSegments : {n : Nat} -> (sizes : Fin n → Nat) -> Fin (segmentedArity sizes) ->
    Σ seg : Fin n, Fin (sizes seg)
  | 0, _, j => nomatch j
  | n + 1, sizes, j =>
      if h : j.1 < sizes 0 then
        ⟨0, ⟨j.1, h⟩⟩
      else
        let jTail : Fin (segmentedArity (fun i : Fin n => sizes i.succ)) :=
          ⟨j.1 - sizes 0, by
            have hle : sizes 0 ≤ j.1 := Nat.le_of_not_lt h
            have hlt0 : j.1 < sizes 0 + segmentedArity (fun i : Fin n => sizes i.succ) := j.2
            have hlt : j.1 < segmentedArity (fun i : Fin n => sizes i.succ) + sizes 0 := by
              exact Nat.lt_of_lt_of_eq hlt0 (Nat.add_comm _ _)
            exact Nat.sub_lt_right_of_lt_add hle hlt⟩
        let ⟨segTail, offTail⟩ := splitSegments (fun i : Fin n => sizes i.succ) jTail
        by
          have off : Fin (sizes segTail.succ) := by
            exact offTail
          exact ⟨segTail.succ, off⟩

/--
N-ary direct sum of remappings. This matches the paper's Definition 14 using a
finite family indexed by `Fin n`.
-/
def directSumFamily {n : Nat} (domSizes codSizes : Fin n → Nat)
    (components : (i : Fin n) → FiniteRemapping (domSizes i) (codSizes i)) :
    FiniteRemapping (segmentedArity domSizes) (segmentedArity codSizes) where
  toFun := fun j =>
    let ⟨seg, off⟩ := splitSegments codSizes j
    joinSegments domSizes seg ((components seg).toFun off)

/-- Pull segment sizes back along a remapping. -/
def pullSegmentSizes {dom cod : Nat} (μ : FiniteRemapping dom cod) (sizes : Fin dom → Nat) :
    Fin cod → Nat :=
  fun j => sizes (μ.toFun j)

/--
Associativity-compatible flat remapping from Definition 15. The remapping acts
on flattened indices by preserving segment-local offsets and transporting the
segment choice through `μ`.
-/
def flatRemapping {dom cod : Nat} (μ : FiniteRemapping dom cod) (sizes : Fin dom → Nat) :
    FiniteRemapping (segmentedArity sizes) (segmentedArity (pullSegmentSizes μ sizes)) where
  toFun := fun j =>
    let ⟨seg, off⟩ := splitSegments (pullSegmentSizes μ sizes) j
    joinSegments sizes (μ.toFun seg) off

theorem split_join :
    {n : Nat} -> (sizes : Fin n → Nat) -> (seg : Fin n) -> (off : Fin (sizes seg)) ->
    splitSegments sizes (joinSegments sizes seg off) = ⟨seg, off⟩
  | 0, _, seg, _ => nomatch seg
  | n + 1, sizes, seg, off => by
      cases seg using Fin.cases with
      | zero =>
          simp [joinSegments, splitSegments]
      | succ segTail =>
          have hnot : ¬ sizes 0 + ↑(joinSegments (fun i : Fin n => sizes i.succ) segTail off) < sizes 0 := by
            exact Nat.not_lt_of_ge (Nat.le_add_right (sizes 0) _)
          have hrec := split_join (fun i : Fin n => sizes i.succ) segTail off
          have htailEq :
              (⟨sizes 0 + ↑(joinSegments (fun i : Fin n => sizes i.succ) segTail off) - sizes 0,
                by
                  have hle : sizes 0 ≤ sizes 0 + ↑(joinSegments (fun i : Fin n => sizes i.succ) segTail off) :=
                    Nat.le_add_right (sizes 0) _
                  have hlt :
                      sizes 0 + ↑(joinSegments (fun i : Fin n => sizes i.succ) segTail off) <
                        segmentedArity (fun i : Fin n => sizes i.succ) + sizes 0 := by
                    have hltLeft :=
                      Nat.add_lt_add_left (joinSegments (fun i : Fin n => sizes i.succ) segTail off).2 (sizes 0)
                    exact Nat.lt_of_lt_of_eq hltLeft (Nat.add_comm _ _)
                  exact Nat.sub_lt_right_of_lt_add hle hlt⟩ :
                Fin (segmentedArity (fun i : Fin n => sizes i.succ))) =
              joinSegments (fun i : Fin n => sizes i.succ) segTail off := by
            apply Fin.ext
            simp [Nat.add_sub_cancel_left]
          simp [joinSegments, splitSegments, hnot, Nat.add_sub_cancel_left]
          constructor
          · simp at hrec ⊢
            exact congrArg Sigma.fst hrec
          · rw [htailEq, hrec]

theorem join_split :
    {n : Nat} -> (sizes : Fin n → Nat) -> (j : Fin (segmentedArity sizes)) ->
    joinSegments sizes (splitSegments sizes j).1 (splitSegments sizes j).2 = j
  | 0, _, j => nomatch j
  | n + 1, sizes, j => by
      by_cases h : j.1 < sizes 0
      · have hsplit : splitSegments sizes j = ⟨0, ⟨j.1, h⟩⟩ := by
          simp [splitSegments, h]
        rw [hsplit]
        apply Fin.ext
        simp [joinSegments]
      · have hle : sizes 0 ≤ j.1 := Nat.le_of_not_lt h
        let jTail : Fin (segmentedArity (fun i : Fin n => sizes i.succ)) :=
          ⟨j.1 - sizes 0, by
            have hlt0 : j.1 < sizes 0 + segmentedArity (fun i : Fin n => sizes i.succ) := j.2
            have hlt : j.1 < segmentedArity (fun i : Fin n => sizes i.succ) + sizes 0 := by
              exact Nat.lt_of_lt_of_eq hlt0 (Nat.add_comm _ _)
            exact Nat.sub_lt_right_of_lt_add hle hlt⟩
        have hsplit : splitSegments sizes j =
            ⟨(splitSegments (fun i : Fin n => sizes i.succ) jTail).fst.succ,
              (splitSegments (fun i : Fin n => sizes i.succ) jTail).snd⟩ := by
          simp [splitSegments, h, jTail]
        have hrec := join_split (fun i : Fin n => sizes i.succ) jTail
        have hnatAdd := congrArg (Fin.natAdd (sizes 0)) hrec
        rw [hsplit]
        simpa [joinSegments, jTail, Nat.add_sub_of_le hle] using hnatAdd

theorem directSumFamily_apply
    {n : Nat} (domSizes codSizes : Fin n → Nat)
    (components : (i : Fin n) → FiniteRemapping (domSizes i) (codSizes i))
    (seg : Fin n) (off : Fin (codSizes seg)) :
    (directSumFamily domSizes codSizes components).toFun (joinSegments codSizes seg off) =
      joinSegments domSizes seg ((components seg).toFun off) := by
  simpa [directSumFamily] using
    congrArg
      (fun p => joinSegments domSizes p.1 ((components p.1).toFun p.2))
      (split_join codSizes seg off)

theorem flatRemapping_join
    {dom cod : Nat} (μ : FiniteRemapping dom cod) (sizes : Fin dom → Nat)
    (seg : Fin cod) (off : Fin ((pullSegmentSizes μ sizes) seg)) :
    (flatRemapping μ sizes).toFun (joinSegments (pullSegmentSizes μ sizes) seg off) =
      joinSegments sizes (μ.toFun seg) off := by
  simpa [flatRemapping] using
    congrArg
      (fun p => joinSegments sizes (μ.toFun p.1) p.2)
      (split_join (pullSegmentSizes μ sizes) seg off)

end FiniteRemapping

/-- Flatten a family of segment sizes into a single arity count. -/
def flattenedArity (segments : List Nat) : Nat :=
  segments.foldl (· + ·) 0

/-- Product block together with input/output remappings. -/
structure RemappedBlock where
  inDom : Nat
  inCod : Nat
  outDom : Nat
  outCod : Nat
  inputRemap : FiniteRemapping inDom inCod
  outputRemap : FiniteRemapping outDom outCod
  base : ProductMorphism

namespace RemappedBlock

def totalBoundarySlots (block : RemappedBlock) : Nat :=
  flattenedArity [block.inDom, block.inCod, block.outDom, block.outCod]

end RemappedBlock

end HeytingLean.Bridge.Abbott.DeepLearningAlgebra
