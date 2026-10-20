// Rasterizer/target configuration: primitive, depthStencil, and
// multisample state, plus the color target's format/blend/writeMask.
// Bound to exactly one Shader at construction time, which supplies the
// vertex buffer and bind group layouts. Builds the immutable
// GPURenderPipeline once, eagerly. Ported from python/pipeline.py.

// Sentinel distinguishing "depthStencil not passed" (use the 3D default
// below) from an explicit null (no depth-stencil attachment at all).
const UNSET = Symbol("unset");

const DEFAULT_PRIMITIVE = { topology: "triangle-list", frontFace: "ccw", cullMode: "none" };
const DEFAULT_MULTISAMPLE = { count: 1, mask: 0xFFFFFFFF, alphaToCoverageEnabled: false };
const DEFAULT_DEPTH_STENCIL = {
  format: "depth24plus", depthWriteEnabled: true, depthCompare: "less",
  depthBias: 0, depthBiasSlopeScale: 0, depthBiasClamp: 0,
};

export class Pipeline {
  // Builds the immutable GPURenderPipeline for `shader`: merges
  // primitive/multisample/depthStencil overrides over the defaults, and
  // derives the fragment stage from `targetFormat` (null for a
  // depth-only pass). Throws if both targetFormat and depthStencil are
  // absent.
  constructor (shader, targetFormat, {
    primitive = null, depthStencil = UNSET, multisample = null,
    blend = null, writeMask = GPUColorWrite.ALL, stencilReference = null,
  } = {}) {
    this.shader = shader;
    this.device = shader.device;
    this.targetFormat = targetFormat;
    this.stencilReference = stencilReference;

    this.primitive = { ...DEFAULT_PRIMITIVE, ...(primitive || {}) };
    this.multisample = { ...DEFAULT_MULTISAMPLE, ...(multisample || {}) };
    if (depthStencil === UNSET) {
      this.depthStencil = { ...DEFAULT_DEPTH_STENCIL };
    } else if (depthStencil === null) {
      this.depthStencil = null;
    } else {
      this.depthStencil = { ...DEFAULT_DEPTH_STENCIL, ...depthStencil };
    }

    // targetFormat=null: a depth-only pipeline (e.g. a shadow-map
    // generation pass) - no color target, no fragment stage at all.
    if (targetFormat === null && this.depthStencil === null) {
      throw new Error("Pipeline needs targetFormat or depthStencil (or both) - neither given");
    }
    let fragment;
    if (targetFormat !== null) {
      const target = { format: targetFormat, writeMask };
      // WebGPU's dictionary members reject an explicit null - "absent"
      // means the key must be omitted entirely, not set to null.
      if (blend !== null) target.blend = blend;
      fragment = {
        module: shader.getModule(), entryPoint: "fs_main",
        targets: [target],
      };
    }

    const vertexBuffers = shader.getVertexBufferLayout();
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: shader.getBindGroupLayouts() });

    const desc = {
      layout: pipelineLayout,
      vertex: { module: shader.getModule(), entryPoint: "vs_main", buffers: vertexBuffers },
      primitive: this.primitive,
      multisample: this.multisample,
    };
    if (this.depthStencil !== null) desc.depthStencil = this.depthStencil;
    if (fragment !== undefined) desc.fragment = fragment;
    this._gpuPipeline = this.device.createRenderPipeline(desc);
  }

  getShader () {
    return this.shader;
  }

  // Binds this Pipeline's GPURenderPipeline and its Shader's "matrix"
  // storage array on the render pass. Called from load (on the way in)
  // and from the unload of a nested Pipeline (on the way out), since the
  // render pass has no stack of its own. Skips the actual rebind if this
  // Pipeline is already the one last bound.
  activate (st) {
    if (st.getActivePipeline() === this) return;
    st.renderPass.setPipeline(this._gpuPipeline);
    st.renderPass.setBindGroup(this.shader.getMatrixGroupIndex(), this.shader.getMatrixBindGroup());
    st.setActivePipeline(this);
  }

  // Pushes this Pipeline onto State's pipeline stack, binds it on the
  // render pass (see activate), delegates to the shader to rewrite and
  // push its "global" block, and applies stencilReference (0 if unset).
  // Pair with unload() around a Node's subtree. activate comes before
  // commitGlobal so bind groups are always set after the pipeline they
  // belong to, never before.
  load (st) {
    st.pushPipeline(this);
    this.activate(st);
    this.shader.commitGlobal(st);
    st.renderPass.setStencilReference(this.stencilReference !== null ? this.stencilReference : 0);
  }

  // Pops this Pipeline off State's stack and restores the enclosing one
  // on the render pass (activate again), restores whatever global bind
  // group was active before this Pipeline's load, and re-applies the
  // enclosing Pipeline's stencilReference (0 if it doesn't set one).
  // Pair with load().
  unload (st) {
    st.popPipeline();
    const top = st.pipeline.length > 0 ? st.pipeline[st.pipeline.length - 1] : null;
    if (top !== null) top.activate(st);
    this.shader.unbindGlobal(st);
    if (top !== null) st.renderPass.setStencilReference(top.stencilReference !== null ? top.stencilReference : 0);
  }
}
