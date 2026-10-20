// Per-render-pass context threaded through Scene.render/Node.render, ported
// from python/state.py: the active camera/device/renderPass, the pipeline
// stack, a per-group bind-group stack (see pushBindGroup/popBindGroup -
// used by Material/TextureSet), and a flat named-value stack per
// "matrix"-group field name (fed by Transform, read by
// Shader.commitMatrix - see pushMatrix/popMatrix/getMatrix).
// "material"/"global"/textures don't go through this value-stack at all -
// each Shader owns its own persistent per-Material buffers, its own
// "global" buffer, and its own per-TextureSet bind groups.
// The "matrix" field name is a stack too, but accumulates via pushMatrix
// instead of a plain push; vertex/normal/projection are derived from it
// once per Node, via loadMatrices/unloadMatrices.
//
// The camera's view/projection matrices are computed once here, not per-
// Node: one State is built fresh per render pass, and the camera is
// logically constant for its whole duration.

import * as gm from "./graphicsmath.js";

export class State {
  // Starts the pipeline/bind-group/value stacks empty and snapshots the
  // camera's view/projection matrices for this whole render pass.
  // cameraPosition isn't seeded here - it's computed directly by
  // Shader.commitGlobal, called via Pipeline.load, since it needs a real
  // Shader active to know the lighting space to compute it in.
  constructor (camera, device, renderPass, canvasSize) {
    this.camera = camera;
    this.device = device;
    this.renderPass = renderPass;
    this.canvasSize = canvasSize;
    this.pipeline = [];
    this._activePipeline = null;
    this.bindGroupStacks = new Map();
    // the only inherited thing that isn't a bind group: the accumulated
    // matrix. A dedicated stack, with no name - this was a generic
    // Map<string, array> that in practice only ever held "matrix".
    this._matrixStack = [];
    this._matrixVersion = 0;
    // shaders that appended a matrix row in this pass, so Renderer /
    // Algorithm know whose array to flush before submit
    this._matrixShaders = [];
    // projection/vertex/normal are not traversal state: they are derived
    // from "matrix" at each drawing Node, read by Shader.commitMatrix in
    // that same Node, then discarded. Plain values, therefore - not stacks.
    this.derived = new Map();

    this._view = camera.getViewMatrix();
    this._proj = camera.getProjMatrix(canvasSize);
    this._viewProj = gm.multiply(this._proj, this._view);
    this._inverseView = null; // lazy - only Shader.commitGlobal's world-space camera_position needs it
  }

  // Pushes `pip` onto the pipeline stack, making it what getPipeline
  // returns until it's popped. Must be paired with popPipeline.
  pushPipeline (pip) {
    this.pipeline.push(pip);
  }

  // Pops the innermost Pipeline off the stack, restoring whatever was
  // active before. Pairs with pushPipeline.
  popPipeline () {
    this.pipeline.pop();
  }

  // Returns the innermost currently-loaded Pipeline (top of the stack);
  // throws if no Node in the current path has set one.
  getPipeline () {
    if (this.pipeline.length === 0) throw new Error("Pipeline not defined");
    return this.pipeline[this.pipeline.length - 1];
  }

  // Returns the shader of the currently active Pipeline.
  getShader () {
    return this.getPipeline().getShader();
  }

  // Returns whichever Pipeline last actually bound its GPU pipeline on
  // the render pass (see setActivePipeline) - null if none has yet.
  getActivePipeline () {
    return this._activePipeline;
  }

  // Records `pip` as the Pipeline whose GPU pipeline/matrix bind group
  // are currently actually bound on the render pass. Called by
  // Pipeline.activate right after it does that binding.
  setActivePipeline (pip) {
    this._activePipeline = pip;
  }

  // Binds `bindGroup` at `groupIndex` immediately and remembers it, so
  // popBindGroup can restore whatever was bound before. Must be paired
  // with popBindGroup.
  pushBindGroup (groupIndex, bindGroup) {
    let stack = this.bindGroupStacks.get(groupIndex);
    if (stack === undefined) { stack = []; this.bindGroupStacks.set(groupIndex, stack); }
    stack.push(bindGroup);
    this.renderPass.setBindGroup(groupIndex, bindGroup);
  }

  // Pops `groupIndex`'s bind-group stack and re-binds whatever's now on
  // top (the ancestor's), if anything remains. Pairs with pushBindGroup.
  popBindGroup (groupIndex) {
    const stack = this.bindGroupStacks.get(groupIndex);
    stack.pop();
    if (stack.length > 0) this.renderPass.setBindGroup(groupIndex, stack[stack.length - 1]);
  }

  // Pushes the current top composed with `mat` - matrix accumulation,
  // where each Transform composes onto its ancestors' matrix rather than
  // overriding it. Identity is the implicit bottom of the stack. Pairs
  // with popMatrix.
  pushMatrix (mat) {
    this._matrixStack.push(gm.multiply(this.getMatrix(), mat));
    this._matrixVersion += 1;
  }

  // Pushes `mat` *without* composing, discarding what the ancestors
  // accumulated - for the rare transform that positions itself in the
  // scene rather than relative to its parent (see SkyBoxTransform).
  pushMatrixAbsolute (mat) {
    this._matrixStack.push(mat);
    this._matrixVersion += 1;
  }

  // Pops the matrix stack, restoring the ancestor's.
  popMatrix () {
    this._matrixStack.pop();
    this._matrixVersion += 1;
  }

  // The accumulated matrix of the path down to here - identity if no
  // Transform pushed anything yet.
  getMatrix () {
    return this._matrixStack.length > 0
      ? this._matrixStack[this._matrixStack.length - 1] : gm.mat4(1.0);
  }

  // Bumped by every push/pop, so Shader.commitMatrix can tell whether the
  // accumulated matrix changed since the row it last handed out.
  getMatrixVersion () {
    return this._matrixVersion;
  }

  // Records that `shader` appended a matrix row in this pass, so
  // Renderer/Algorithm know whose array to flush before submit.
  registerMatrixShader (shader) {
    if (!this._matrixShaders.includes(shader)) this._matrixShaders.push(shader);
  }

  getMatrixShaders () {
    return this._matrixShaders;
  }

  getCamera () {
    return this.camera;
  }

  // Returns this render pass's camera view matrix, computed once in the constructor.
  getViewMatrix () {
    return this._view;
  }

  // Returns this render pass's camera projection matrix, computed once in the constructor.
  getProjMatrix () {
    return this._proj;
  }

  // Returns the precomputed proj * view matrix, once per render pass.
  getViewProjMatrix () {
    return this._viewProj;
  }

  // Returns the inverse view matrix, computed on first use and cached -
  // only Shader.commitGlobal's world-space camera_position needs it.
  getInverseViewMatrix () {
    if (this._inverseView === null) this._inverseView = gm.inverse(this._view);
    return this._inverseView;
  }

  // Computes projection/vertex/normal from the current "matrix" stack and
  // this render pass's cached camera matrices (respecting the active
  // shader's lighting space) and stores them in `derived`. Called once per
  // Node just before its Shapes are drawn, paired with unloadMatrices right
  // after: they are freshly derived per Node and never inherited by
  // descendants, so they are plain values, not stacks - only "matrix" needs
  // stack discipline. vertex/normal are skipped entirely if the active
  // shader's "matrix" struct doesn't declare them.
  loadMatrices () {
    // "projection" is deliberately NOT here: it takes the lighting space to
    // NDC and is the same for the whole pass, so it belongs to the scene,
    // not to a node - Shader.commitGlobal writes it into the "global" block.
    const shd = this.getShader();
    const mat = this.getMatrix();
    const needsVertex = shd.declaresMatrixField("vertex");
    const needsNormal = shd.declaresMatrixField("normal");
    if (needsVertex || needsNormal) {
      let mv = mat; // to global space
      if (shd.getLightingSpace() === "camera") mv = gm.multiply(this.getViewMatrix(), mv); // to camera space
      if (needsVertex) this.derived.set("vertex", mv);
      if (needsNormal) {
        // A rank-deficient transform (e.g. the planar-shadow projection,
        // which flattens geometry onto a plane) has no inverse; the pass
        // that uses one never lights anything, so identity is a safe
        // stand-in and beats throwing in the middle of a frame.
        try {
          this.derived.set("normal", gm.transpose(gm.inverse(mv)));
        } catch {
          this.derived.set("normal", gm.mat4(1.0));
        }
      }
    }
  }

  // Discards what loadMatrices derived. Pairs with it. Nothing is restored:
  // no descendant ever inherits these - the next Node that draws derives its
  // own from its own accumulated "matrix".
  unloadMatrices () {
    this.derived.clear();
  }

  // The matrix loadMatrices derived for `name` in the Node being drawn, or
  // null if this shader doesn't declare it. Read by Shader.commitMatrix.
  getDerived (name) {
    const v = this.derived.get(name);
    return v === undefined ? null : v;
  }
}
