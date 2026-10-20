// Sampling rule (addressing/filter), independent of any Texture class. A
// pure resource-holder - no load/unload of its own, see TextureSet.
// Ported from python/sampler.py.

export class Sampler {
  // Creates a regular filtering sampler, or a comparison sampler (usable
  // with texture_depth types, e.g. shadow map PCF) when `compare` is given.
  constructor (device, varname, {
    addressModeU = "repeat", addressModeV = "repeat",
    magFilter = "linear", minFilter = "linear", mipmapFilter = "linear",
    compare = null,
  } = {}) {
    this.varname = varname;
    const desc = { addressModeU, addressModeV, magFilter, minFilter, mipmapFilter };
    if (compare !== null) desc.compare = compare;
    this.sampler = device.createSampler(desc);
  }

  // The GPU resource TextureSet/Shader.addTextureSet binds - this sampler
  // itself.
  get resource () {
    return this.sampler;
  }
}
