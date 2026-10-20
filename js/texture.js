// 2D texture bound to `varname`, stored as rgba8unorm-srgb. A pure
// resource-holder - no load/unload of its own, see TextureSet. Ported
// from python/texture.py.
//
// (s,t) = (0,0) is the top-left corner, per WebGPU convention - no
// vertical flip on load. All geometry in this repo (Disk, Square, Quad,
// Cube, Sphere) generates t increasing downward, accordingly.

export class Texture {
  // Builds a solid-color (`texel`) or blank (`width` x `height`) texture.
  // For an image file, use Texture.load instead - decoding an image is
  // async in the browser, unlike python's synchronous PIL.Image.open.
  // format: usar "rgba8unorm" para mapa de normais. O padrao sRGB faz a GPU
  // desfazer a correcao de gama na leitura, o que e certo para cores mas
  // distorce os vetores guardados num mapa de normais.
  constructor (device, varname, { texel = null, width = 1, height = 1, format = "rgba8unorm-srgb" } = {}) {
    this.varname = varname;
    let data;
    if (texel === null) {
      data = new Uint8Array(width * height * 4);
    } else if (texel.length === 3) {
      width = 1; height = 1;
      data = new Uint8Array([texel[0] * 255, texel[1] * 255, texel[2] * 255, 255]);
    } else if (texel.length === 4) {
      width = 1; height = 1;
      data = new Uint8Array([texel[0] * 255, texel[1] * 255, texel[2] * 255, texel[3] * 255]);
    } else {
      throw new Error("Invalid Texture parameters");
    }
    this.width = width;
    this.height = height;

    // "-srgb": the bytes of an image (jpg/png) already come sRGB-encoded
    // by convention - the GPU undoes that curve when sampling, delivering
    // the shader a true linear value.
    this.tex = device.createTexture({
      size: [width, height, 1], format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this.tex }, data,
      { bytesPerRow: width * 4, rowsPerImage: height }, [width, height, 1],
    );
    this.view = this.tex.createView();
  }

  // Fetches and decodes the image at `url`, uploading it as a texture
  // bound to `varname`.
  // format: usar "rgba8unorm" para mapa de normais (sem conversao sRGB)
  static async load (device, varname, url, { format = "rgba8unorm-srgb" } = {}) {
    const blob = await (await fetch(url)).blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
    const texture = Object.create(Texture.prototype);
    texture.varname = varname;
    texture.width = bitmap.width;
    texture.height = bitmap.height;
    texture.tex = device.createTexture({
      size: [bitmap.width, bitmap.height, 1], format,
      // RENDER_ATTACHMENT is required by copyExternalImageToTexture's destination.
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
      { source: bitmap, flipY: false }, { texture: texture.tex }, [bitmap.width, bitmap.height, 1],
    );
    texture.view = texture.tex.createView();
    return texture;
  }

  getTexture () {
    return this.tex;
  }

  getWidth () {
    return this.width;
  }

  getHeight () {
    return this.height;
  }

  // The GPU resource TextureSet/Shader.addTextureSet binds - this
  // texture's view.
  get resource () {
    return this.view;
  }
}
