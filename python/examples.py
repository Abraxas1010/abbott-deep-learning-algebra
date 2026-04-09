from __future__ import annotations

import numpy as np

try:
    from .json_schema import (
        ArrayObjectSchema,
        BroadcastedOperationSchema,
        ReindexingSchema,
        slot_ref,
    )
    from .uid_config import AxisSpec, UIDConfig
except ImportError:
    from json_schema import (
        ArrayObjectSchema,
        BroadcastedOperationSchema,
        ReindexingSchema,
        slot_ref,
    )
    from uid_config import AxisSpec, UIDConfig


def convolution_example() -> BroadcastedOperationSchema:
    batch = AxisSpec(uid=0, label="batch", size=8)
    channel = AxisSpec(uid=1, label="channel", size=16)
    length = AxisSpec(uid=2, label="length", size=64)
    kernel_width = AxisSpec(uid=3, label="kernel_width", size=3)
    signal = ArrayObjectSchema(dtype="float", axes=[batch, channel, length])
    kernel = ArrayObjectSchema(dtype="float", axes=[channel, kernel_width])
    config = UIDConfig(
        slots=[
            slot_ref(0, "axis", "batch"),
            slot_ref(1, "axis", "channel"),
            slot_ref(2, "axis", "length"),
            slot_ref(3, "axis", "kernel_width"),
            slot_ref(10, "op_param", "kernel_width"),
            slot_ref(11, "op_param", "padding_mode"),
        ],
        axes=[batch, channel, length, kernel_width],
        dtypes=["float"],
        op_params={10: 3, 11: "circular"},
    )
    return BroadcastedOperationSchema(
        name="convolution",
        inputs=[signal, kernel],
        output=signal,
        uid_config=config,
        reindexings=[
            ReindexingSchema(
                source_axes=[0, 1, 2],
                target_axes=[0, 1, 2, 3],
                linear=[
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                ],
                offset=[0, 0, -1],
            )
        ],
        input_weaves=[[0, 1, 2, 3]],
        output_weaves=[[0, 1, 2]],
    )


def attention_example() -> BroadcastedOperationSchema:
    token = AxisSpec(uid=10, label="token", size=128)
    feature = AxisSpec(uid=11, label="feature", size=64)
    obj = ArrayObjectSchema(dtype="float", axes=[token, feature])
    config = UIDConfig(
        slots=[
            slot_ref(10, "axis", "token"),
            slot_ref(11, "axis", "feature"),
            slot_ref(20, "op_param", "heads"),
        ],
        axes=[token, feature],
        dtypes=["float"],
        op_params={20: 8},
    )
    return BroadcastedOperationSchema(
        name="self_attention",
        inputs=[obj, obj, obj],
        output=obj,
        uid_config=config,
        input_weaves=[[0, 1]],
        output_weaves=[[0, 1]],
    )


def convolution_sample_tensors() -> tuple[np.ndarray, np.ndarray]:
    schema = convolution_example()
    signal_shape = tuple(axis.size for axis in schema.inputs[0].axes)
    kernel_shape = tuple(axis.size for axis in schema.inputs[1].axes)
    signal = np.arange(np.prod(signal_shape), dtype=np.float64).reshape(signal_shape)
    kernel = np.linspace(1.0, 2.0, num=np.prod(kernel_shape), dtype=np.float64).reshape(kernel_shape)
    return signal, kernel


def attention_sample_tensors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    schema = attention_example()
    shape = tuple(axis.size for axis in schema.inputs[0].axes)
    base = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    query = base / 100.0
    key = (base + 1.0) / 120.0
    value = (base + 2.0) / 140.0
    return query, key, value
