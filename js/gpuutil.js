// Small shared helper used by every Shape: creates a GPUBuffer already
// filled with `data` (a TypedArray), mirroring wgpu-py's
// device.create_buffer_with_data convenience - raw WebGPU JS has no
// single-call equivalent, only createBuffer({mappedAtCreation}) + write +
// unmap.
export function createBufferWithData (device, data, usage) {
  const size = Math.ceil(data.byteLength / 4) * 4;
  const buffer = device.createBuffer({ size, usage, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}
