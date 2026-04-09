export type SlotKind = "axis" | "dtype" | "op_param" | "placeholder";

export interface SlotRef {
  uid: number;
  kind: SlotKind;
  label?: string;
}

export interface AxisSpec {
  uid: number;
  label: string;
  size: number;
}

export interface ArrayObjectSchema {
  dtype: string;
  axes: AxisSpec[];
}

export interface ReindexingSchema {
  source_axes: number[];
  target_axes: number[];
  linear: number[][];
  offset: number[];
}

export interface BroadcastedOperationSchema {
  name: string;
  inputs: ArrayObjectSchema[];
  output: ArrayObjectSchema;
  uid_config: {
    slots: SlotRef[];
    axes: AxisSpec[];
    dtypes: string[];
    op_params: Record<number, string | number>;
  };
  reindexings: ReindexingSchema[];
  input_weaves: number[][];
  output_weaves: number[][];
}
