// Manual WGSL uniform/storage buffer layout + packing, ported from
// python/uniformbuffer.py. Implements WGSL's layout rules by hand
// (WebGPU's std140/std430-equivalent): each field is placed at the next
// offset that's a multiple of its own type's alignment (not just its size
// - e.g. vec3 is 12 bytes but aligns to 16, leaving a 4-byte hole after it
// whenever something follows it), fields are laid out in declaration order
// with no reordering/repacking, and the overall element size is rounded up
// to a multiple of 16 bytes.

import * as gm from "./graphicsmath.js";

// [align, size] in bytes.
const ALIGN_SIZE = {
  i32: [4, 4],
  f32: [4, 4],
  vec2: [8, 8],
  vec3: [16, 12],
  vec4: [16, 16],
  mat4x4: [16, 64],
};

const COMPONENT_COUNT = { vec2: 2, vec3: 3, vec4: 4, mat4x4: 16 };

function zeroFor (wgslType) {
  switch (wgslType) {
    case "i32": return 0;
    case "f32": return 0.0;
    case "vec2": return gm.vec2(0);
    case "vec3": return gm.vec3(0);
    case "vec4": return gm.vec4(0);
    case "mat4x4": return gm.mat4(0);
    default: throw new TypeError("Unsupported wgsl_type in uniformbuffer: " + wgslType);
  }
}

function align (offset, alignment) {
  return Math.ceil(offset / alignment) * alignment;
}

function pack (value, wgslType) {
  if (wgslType === "i32") {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, Math.trunc(value), true);
    return new Uint8Array(buf);
  }
  if (wgslType === "f32") {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    return new Uint8Array(buf);
  }
  if (wgslType === "mat4x4") {
    return new Uint8Array(gm.mat4Bytes(value).buffer);
  }
  if (wgslType in COMPONENT_COUNT) {
    const n = COMPONENT_COUNT[wgslType];
    const vec = value instanceof Float32Array ? value : Float32Array.from(value);
    if (vec.length !== n) throw new Error(`${wgslType} value must have length ${n}, got ${vec.length}`);
    return new Uint8Array(vec.buffer, vec.byteOffset, vec.byteLength);
  }
  throw new TypeError("Unsupported wgsl_type in uniformbuffer: " + wgslType);
}

// [align, size] for one field's type: a direct table lookup for a
// primitive, or - recursively, for a nested struct schema - align is the
// max alignment among its own members and size is that alignment rounded
// up from the unrounded end of its last member (WGSL's own struct layout
// rule; distinct from the 16-byte rounding computeLayout applies to the
// outermost buffer/stride, which nothing but the top level gets).
function typeAlignSize (wgslType) {
  if (Array.isArray(wgslType)) {
    const { naturalSize, align: a } = layoutSchema(wgslType);
    return [a, align(naturalSize, a)];
  }
  return ALIGN_SIZE[wgslType];
}

// Lays out `schema`'s own fields starting at offset 0, recursing into any
// nested struct field so the returned offsets/types are flat, with a
// dotted name ("light.color") for every leaf reached through a nested
// struct. Returns {offsets, types, naturalSize, align}: naturalSize is the
// unrounded offset just past the last member (rounding it to this
// struct's own alignment, or to 16 for the outermost buffer, is the
// caller's job); align is the max alignment among schema's own members.
function layoutSchema (schema) {
  const offsets = {};
  const types = {};
  let offset = 0;
  let a = 1;
  for (const [name, wgslType] of schema) {
    const [fieldAlign, size] = typeAlignSize(wgslType);
    a = Math.max(a, fieldAlign);
    offset = align(offset, fieldAlign);
    if (Array.isArray(wgslType)) {
      const sub = layoutSchema(wgslType);
      for (const subName in sub.offsets) {
        offsets[`${name}.${subName}`] = offset + sub.offsets[subName];
        types[`${name}.${subName}`] = sub.types[subName];
      }
    } else {
      offsets[name] = offset;
      types[name] = wgslType;
    }
    offset += size;
  }
  return { offsets, types, naturalSize: offset, align: a };
}

// Computes every leaf field's byte offset and WGSL type (flat, with a
// dotted name for one nested inside a struct field) and the overall
// (16-byte-rounded) element size - shared by StorageArray (one element's
// stride) and UniformBlock (the whole buffer's size).
function computeLayout (schema) {
  const { offsets, types, naturalSize } = layoutSchema(schema);
  return { offsets, types, size: align(naturalSize, 16) };
}

// Packs `values` (missing fields default to a type-appropriate zero) into
// one `stride`-sized row, fields placed at their precomputed offsets.
// `types` is the flat, dotted-leaf mapping computeLayout returns.
function packRow (values, types, offsets, stride) {
  const row = new Uint8Array(stride);
  for (const name in types) {
    const wgslType = types[name];
    const value = name in values ? values[name] : zeroFor(wgslType);
    const data = pack(value, wgslType);
    row.set(data, offsets[name]);
  }
  return row;
}

// A growable `var<storage, read> name: array<StructType>` buffer - used
// for the "matrix" group, where every draw call needs its own row (many
// transforms per frame, appended rather than overwritten so writeBuffer
// timing relative to a single per-frame submit() doesn't matter - see
// Shader.commitMatrix). Pre-allocated to `maxRows`; append throws if that
// capacity is exceeded rather than silently corrupting or growing
// unbounded.
export class StorageArray {
  constructor (device, schema, layout, groupIndex, maxRows, binding = 0) {
    this.device = device;
    this.schema = schema;
    this.groupIndex = groupIndex;
    this.maxRows = maxRows;
    const computed = computeLayout(schema);
    this.offsets = computed.offsets;
    this._types = computed.types;
    this.stride = computed.size;
    this.count = 0;
    this.buffer = device.createBuffer({
      size: this.stride * maxRows,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.bindGroup = device.createBindGroup({
      layout,
      entries: [{ binding, resource: { buffer: this.buffer, offset: 0, size: this.stride * maxRows } }],
    });
  }

  // Packs `values` into the next free row and writes it, returning that
  // row's index (the value to pass as firstInstance).
  append (values) {
    if (this.count >= this.maxRows) {
      throw new Error(
        `StorageArray for group ${this.groupIndex} exceeded its ${this.maxRows}-row capacity ` +
        "(see Shader's maxInstances) - raise it or draw fewer distinct instances per frame"
      );
    }
    const row = this.count;
    const packed = packRow(values, this._types, this.offsets, this.stride);
    if (this._staging === undefined) this._staging = new Uint8Array(this.maxRows * this.stride);
    this._staging.set(new Uint8Array(packed.buffer ?? packed), row * this.stride);
    this.count += 1;
    return row;
  }

  // Uploads every row appended since the last reset in one writeBuffer.
  // Must run before the pass's submit - anywhere before it works, since
  // writeBuffer is on the queue timeline. No-op when nothing was appended.
  flush () {
    if (this.count === 0) return;
    this.device.queue.writeBuffer(this.buffer, 0,
      this._staging.subarray(0, this.count * this.stride));
  }

  // Starts a fresh frame: every previously-appended row becomes
  // unreachable and capacity is reclaimed from row 0.
  reset () {
    this.count = 0;
  }
}

// A single-instance `var<uniform> name: StructType` buffer + bind group,
// built once and reused for its owner's whole lifetime - Shader builds one
// per registered Material (see addMaterial) and one for its own "global"
// group (see commitGlobal), both eagerly, so createBuffer/createBindGroup
// never happen during traversal.
//
// Written via begin/set/end: set(name, value) throws on a name this
// block's schema doesn't declare - callers that want to skip an optional
// field check hasField first. A field nested inside a struct-typed field
// is named with a dot ("light.color").
export class UniformBlock {
  constructor (device, schema, layout, groupIndex, binding = 0) {
    this.device = device;
    this.schema = schema;
    this.groupIndex = groupIndex;
    const computed = computeLayout(schema);
    this.offsets = computed.offsets;
    this._types = computed.types;
    this.size = computed.size;
    this._pending = null;
    this.buffer = device.createBuffer({
      size: this.size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.bindGroup = device.createBindGroup({
      layout,
      entries: [{ binding, resource: { buffer: this.buffer, offset: 0, size: this.size } }],
    });
  }

  hasField (name) {
    return name in this.offsets;
  }

  // Starts a fresh write pass: a zeroed staging row, so any field never
  // set this pass stays zero. Pairs with end.
  begin () {
    this._pending = new Uint8Array(this.size);
  }

  // Packs `value` into field `name` of the row started by begin. Throws
  // if `name` isn't declared in this block's schema.
  set (name, value) {
    if (!(name in this.offsets)) throw new Error(`'${name}' is not a field of this shader's group`);
    const data = pack(value, this._types[name]);
    this._pending.set(data, this.offsets[name]);
  }

  // Writes the row accumulated since begin to the GPU buffer in one call.
  end () {
    this.device.queue.writeBuffer(this.buffer, 0, this._pending);
    this._pending = null;
  }
}
