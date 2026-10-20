// Wraps a compiled WGSL module: vertex input layout, lighting (light/
// space), and bind groups discovered via reflection. Owns everything
// related to shading - materials, "global" uniforms, and textures/
// samplers - each created and validated once, eagerly, at setup (see
// addMaterial/commitGlobal/addTextureSet below), never during graph
// traversal. Rasterizer state lives in Pipeline, not here. Ported from
// python/shader.py.
//
// Four different strategies for the four conventional groups:
// - "matrix": a growable StorageArray, one row appended per distinct
//   value, indexed per-draw via instanceIndex (see commitMatrix).
// - "material": one persistent UniformBlock per registered Material
//   instance (see addMaterial/bindMaterial/unbindMaterial) - a Material
//   must be added to every shader it's used under; using one that wasn't
//   throws, and a shader with no "material" group at all is always a
//   no-op regardless.
// - "global": one persistent UniformBlock, built here at construction,
//   fed by setValue/getValue plus camera position and the shader's own
//   `light` (see commitGlobal).
// - textures/samplers: one persistent GPUBindGroup per registered
//   TextureSet (see addTextureSet/bindTextureSet/unbindTextureSet).

import * as gm from "./graphicsmath.js";
import * as sutl from "./shaderutl.js";
import { ShaderReflection } from "./wgslreflect.js";
import { StorageArray, UniformBlock } from "./uniformbuffer.js";

export class Shader {
  // Wraps an already-compiled WGSL `code` string. Prefer Shader.load,
  // which fetches `wgslPath` first - the browser has no synchronous file
  // read, unlike python's sutl.readfile.
  constructor (device, code, { light = null, space = "camera", maxInstances = 1024 } = {}) {
    this.device = device;
    this.light = light;
    this.space = space;

    this.module = device.createShaderModule({ code });
    this._reflection = new ShaderReflection(code);

    this._layouts = new Map();
    for (const g of this._reflection.groupIndices()) {
      this._layouts.set(g, device.createBindGroupLayout({ entries: this._reflection.layoutEntries(g) }));
    }

    this._matrixArray = null;
    const matrixGroup = this._tryUniformVarGroup("matrix");
    if (matrixGroup !== null) {
      const fields = this._reflection.storageArrayFields(matrixGroup);
      this._matrixArray = new StorageArray(device, fields, this._layouts.get(matrixGroup), matrixGroup, maxInstances);
    }
    this._frameState = null;
    this._matrixTick = -1;
    this._matrixRow = -1;

    // --- "material" group: discovered here, buffers built lazily per
    // Material instance via addMaterial (never here) ---
    this._materialGroup = this._tryUniformVarGroup("material");
    this._materials = new Map();
    this._materialRevisions = new Map();

    // --- "global" group: built eagerly, once, right here - never rebuilt
    // per Pipeline.load (see commitGlobal) ---
    const globalGroup = this._tryUniformVarGroup("global");
    this._globalBlock = null;
    if (globalGroup !== null) {
      const fields = this._reflection.uniformFields(globalGroup);
      this._globalBlock = new UniformBlock(device, fields, this._layouts.get(globalGroup), globalGroup);
    }
    this._values = new Map();

    // --- texture/sampler group: every texture/sampler this shader needs
    // lives under one @group - discovered as the one declared group with
    // neither uniformFields nor storageArrayFields ---
    const textureGroups = this._reflection.groupIndices().filter(
      (g) => this._reflection.uniformFields(g) === null && this._reflection.storageArrayFields(g) === null
    );
    if (textureGroups.length > 1) {
      throw new Error(
        `shader declares more than one texture/sampler-only group ${textureGroups} - ` +
        "merge them into a single @group"
      );
    }
    this._textureGroup = textureGroups.length > 0 ? textureGroups[0] : null;
    this._textureSets = new Map();

    // vertex buffers registered via setVertexBuffers - see getVertexBufferLayout
    this._vertexBuffers = [];
  }

  // Fetches `wgslPath` and constructs a Shader from it.
  static async load (device, wgslPath, options = {}) {
    const code = await sutl.readfile(wgslPath);
    return new Shader(device, code, options);
  }

  _tryUniformVarGroup (varname) {
    try {
      return this._reflection.uniformVarGroup(varname);
    } catch {
      return null;
    }
  }

  // --- "matrix" storage array: one row per distinct value, indexed per-draw ---

  // Called once per Shape.draw(): resolves the current "vertex"/"normal"/
  // "projection" values to a StorageArray row (packing a fresh row only
  // when the "matrix" group's version has changed since the last commit
  // this frame - see State.getMatrixVersion), and returns that row's
  // index. Pass it as firstInstance to draw()/drawIndexed() - vs_main's
  // @builtin(instance_index) reads it back.
  commitMatrix (st) {
    if (this._matrixArray === null) {
      throw new Error('commitMatrix called on a shader with no "matrix" storage array declared');
    }
    if (st !== this._frameState) {
      this._frameState = st;
      this._matrixArray.reset();
      this._matrixTick = -1;
      this._matrixRow = -1;
    }
    st.registerMatrixShader(this);
    const latest = st.getMatrixVersion();
    if (latest !== this._matrixTick) {
      const values = {};
      for (const [name] of this._matrixArray.schema) {
        const v = st.getDerived(name);
        if (v !== null) values[name] = v;
      }
      this._matrixRow = this._matrixArray.append(values);
      this._matrixTick = latest;
    }
    return this._matrixRow;
  }

  // Returns the "matrix" StorageArray's bind group, for Pipeline.activate
  // to (re)bind unconditionally every time.
  getMatrixBindGroup () {
    if (this._matrixArray === null) {
      throw new Error('getMatrixBindGroup called on a shader with no "matrix" storage array declared');
    }
    return this._matrixArray.bindGroup;
  }

  // Returns the @group index this shader's "matrix" StorageArray is bound
  // at, for Pipeline.activate to bind it at the right slot.
  getMatrixGroupIndex () {
    if (this._matrixArray === null) {
      throw new Error('getMatrixGroupIndex called on a shader with no "matrix" storage array declared');
    }
    return this._matrixArray.groupIndex;
  }

  // Returns whether this shader's "matrix" StorageArray schema declares a
  // field named `name` (e.g. "normal") - used by State.loadMatrices/
  // unloadMatrices to skip computing and pushing a value this shader's
  // own commitMatrix would never read anyway. False if this shader has
  // no "matrix" group at all.
  declaresMatrixField (name) {
    if (this._matrixArray === null) return false;
    return this._matrixArray.schema.some(([n]) => n === name);
  }

  // --- "material" group: one persistent UniformBlock per registered Material ---

  // Registers `mat` with this shader: builds its persistent UniformBlock
  // and writes its initial fields, right here (never during traversal).
  // Throws if this shader has no "material" group. Must be called once,
  // at setup, before `mat` is ever used under this shader.
  addMaterial (mat) {
    if (this._materialGroup === null) throw new Error('this shader has no "material" group');
    const fields = this._reflection.uniformFields(this._materialGroup);
    const block = new UniformBlock(this.device, fields, this._layouts.get(this._materialGroup), this._materialGroup);
    block.begin();
    mat.writeFields(block);
    block.end();
    this._materials.set(mat, block);
    this._materialRevisions.set(mat, mat.getRevision());
  }

  // Binds `mat`'s persistent bind group, rewriting it first if this
  // shader's own copy is behind `mat`'s current revision - tracked per
  // shader (not a shared flag on `mat`). No-op if this shader has no
  // "material" group at all; throws if it does but `mat` was never
  // addMaterial()'d here.
  bindMaterial (st, mat) {
    if (this._materialGroup === null) return;
    const block = this._materials.get(mat);
    if (block === undefined) throw new Error("Material was used under a shader it was never addMaterial()'d to");
    if (this._materialRevisions.get(mat) !== mat.getRevision()) {
      block.begin();
      mat.writeFields(block);
      block.end();
      this._materialRevisions.set(mat, mat.getRevision());
    }
    st.pushBindGroup(block.groupIndex, block.bindGroup);
  }

  // Restores whatever material bind group was active before the matching
  // bindMaterial call. No-op if this shader has no "material" group.
  unbindMaterial (st) {
    if (this._materialGroup === null) return;
    st.popBindGroup(this._materialGroup);
  }

  // --- "global" group: one persistent UniformBlock, fed by setValue + camera + light ---

  // Sets the "global" group field `name` to `value` - the entry point for
  // clip plane, fog, or any other app-global. Throws the next
  // commitGlobal() if `name` isn't a field this shader's "global" struct
  // declares.
  setValue (name, value) {
    this._values.set(name, value);
  }

  getValue (name, defaultValue = null) {
    return this._values.has(name) ? this._values.get(name) : defaultValue;
  }

  // Camera position in this shader's lighting space - in camera space the
  // camera is always at the origin. Uses State's cached, lazily-inverted
  // view matrix rather than recomputing it here.
  _computeCameraPosition (st) {
    let cameraPosition = gm.vec4(0, 0, 0, 1);
    if (this.getLightingSpace() === "world") {
      const mat = st.getInverseViewMatrix();
      cameraPosition = gm.mat4MulVec4(mat, cameraPosition);
    }
    return cameraPosition;
  }

  // Rewrites and pushes this shader's "global" block, once per
  // Pipeline.load: cameraPosition and this shader's own `light`
  // contribution are attempted but skipped if this particular shader's
  // "global" struct doesn't declare them; every app-set setValue is
  // written unconditionally (throws if undeclared). Always rewrites, no
  // dirty-check. No-op if this shader has no "global" group. Pushed via
  // State's bind-group stack, restored by unbindGlobal on the matching
  // Pipeline.unload.
  // The matrix taking this shader's lighting space to NDC - the scene's
  // half of the old per-instance MVP. In camera space the view transform is
  // already folded into "vertex", so only the projection remains; in world
  // space it is projection * view.
  _computeProjectionMatrix (st) {
    return this.getLightingSpace() === "camera"
      ? st.getProjMatrix() : st.getViewProjMatrix();
  }

  // Uploads this pass's matrix rows in one transfer; called by
  // Renderer/Algorithm after the traversal, before submit.
  flushMatrices () {
    if (this._matrixArray !== null) this._matrixArray.flush();
  }

  commitGlobal (st) {
    const block = this._globalBlock;
    if (block === null) return;
    block.begin();
    if (block.hasField("projection")) {
      block.set("projection", this._computeProjectionMatrix(st));
    }
    if (block.hasField("camera_position")) {
      block.set("camera_position", this._computeCameraPosition(st));
    }
    if (this.light !== null) {
      this.light.writeFields(block, st, this.getLightingSpace());
    }
    for (const [name, value] of this._values) block.set(name, value);
    block.end();
    st.pushBindGroup(block.groupIndex, block.bindGroup);
  }

  // Restores whatever global bind group was active before the matching
  // commitGlobal call. No-op if this shader has no "global" group.
  unbindGlobal (st) {
    if (this._globalBlock === null) return;
    st.popBindGroup(this._globalBlock.groupIndex);
  }

  // --- textures/samplers: one persistent GPUBindGroup per registered TextureSet ---

  // Returns [group, binding] for the texture or sampler variable
  // `varname`; throws if this shader declares neither.
  _resolveBinding (varname) {
    try {
      return this._reflection.textureBinding(varname);
    } catch {
      return this._reflection.samplerBinding(varname);
    }
  }

  // Registers `ts` with this shader: validates it covers this shader's
  // texture/sampler group exactly and builds its persistent bind group,
  // right here (never during traversal). Throws if this shader has no
  // texture/sampler group, if an item's varname isn't declared by this
  // shader, or if the set is incomplete/has a duplicate binding. Retains
  // every item that supports it (e.g. Texture1D).
  addTextureSet (ts) {
    if (this._textureGroup === null) throw new Error("this shader has no texture/sampler group");
    const entries = [];
    const seen = new Set();
    for (const item of ts.items) {
      let group, binding;
      try {
        [group, binding] = this._resolveBinding(item.varname);
      } catch {
        throw new Error(`'${item.varname}' is not declared by this shader`);
      }
      if (group !== this._textureGroup) throw new Error(`'${item.varname}' does not belong to this shader's texture group`);
      if (seen.has(binding)) throw new Error(`binding ${binding} supplied more than once`);
      seen.add(binding);
      entries.push({ binding, resource: item.resource });
    }
    const expected = this._reflection.layoutEntries(this._textureGroup).length;
    if (entries.length !== expected) throw new Error(`texture set covers ${entries.length} binding(s), shader declares ${expected}`);
    this._textureSets.set(ts, this.device.createBindGroup({ layout: this._layouts.get(this._textureGroup), entries }));
    for (const item of ts.items) {
      if (typeof item.retain === "function") item.retain();
    }
  }

  // Binds `ts`'s persistent bind group. No-op if this shader has no
  // texture/sampler group at all; throws if it does but `ts` was never
  // addTextureSet()'d here.
  bindTextureSet (st, ts) {
    if (this._textureGroup === null) return;
    const bindGroup = this._textureSets.get(ts);
    if (bindGroup === undefined) throw new Error("TextureSet was used under a shader it was never addTextureSet()'d to");
    st.pushBindGroup(this._textureGroup, bindGroup);
  }

  // Restores whatever texture bind group was active before the matching
  // bindTextureSet call. No-op if this shader has no texture/sampler group.
  unbindTextureSet (st) {
    if (this._textureGroup === null) return;
    st.popBindGroup(this._textureGroup);
  }

  // --- used by State, to resolve a pushed field name to its group ---


  // --- used by Pipeline, to build the GPURenderPipeline ---

  // Returns the compiled GPUShaderModule, for Pipeline's vertex and
  // fragment stages.
  getModule () {
    return this.module;
  }

  // Layouts in group order (0,1,2,...) - required by createPipelineLayout.
  getBindGroupLayouts () {
    return this._reflection.groupIndices().map((g) => this._layouts.get(g));
  }

  // Returns the coordinate space ("world" or "camera") this shader
  // expects the light's position in; Light.writeFields uses this to
  // decide how to transform light_position before writing it.
  getLightingSpace () {
    return this.space;
  }

  // --- vertex input contract: the app declares how its buffers are packed ---

  // Registers this shader's vertex buffer layout, in WebGPU's
  // vertex.buffers shape; any attribute may give `varName` instead of
  // `shaderLocation`, resolved to that input's declared @location. Throws
  // if a `varName` isn't a declared vertex input.
  setVertexBuffers (buffers) {
    const resolved = [];
    for (const buf of buffers) {
      const attributes = [];
      for (const attr of buf.attributes) {
        const a = { ...attr };
        if ("varName" in a) {
          const name = a.varName;
          delete a.varName;
          try {
            a.shaderLocation = this._reflection.vertexLocation(name);
          } catch {
            throw new Error(`setVertexBuffers references '${name}', but this shader doesn't declare a vertex input with that name`);
          }
        }
        attributes.push(a);
      }
      resolved.push({ ...buf, attributes });
    }
    this._vertexBuffers = resolved;
  }

  // Returns the registered vertex.buffers list, after checking every
  // @location this shader declares is covered by exactly one attribute.
  // Throws on a mismatch.
  getVertexBufferLayout () {
    const covered = new Set();
    for (const buf of this._vertexBuffers) {
      for (const attr of buf.attributes) {
        const location = attr.shaderLocation;
        if (covered.has(location)) throw new Error(`Vertex buffer location ${location} registered more than once via setVertexBuffers`);
        covered.add(location);
      }
    }
    const missing = [...this._reflection.vertexLocations()].filter((l) => !covered.has(l));
    if (missing.length > 0) {
      throw new Error(`Shader declares vertex input location(s) ${missing.sort((a, b) => a - b)} with no matching setVertexBuffers attribute`);
    }
    return this._vertexBuffers;
  }
}
