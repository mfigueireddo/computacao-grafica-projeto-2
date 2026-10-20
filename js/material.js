// Generic named-value "material" appearance plus a revision counter.
// Ported from python/material.py. It can be used directly for an
// arbitrary shader material block, or as the base of a semantic
// convenience class such as PhongMaterial.
//
// Material has no Shader/GPU knowledge. Each Shader owns the actual
// UniformBlock *and* tracks the last revision it uploaded (see
// Shader.addMaterial/bindMaterial) - two different shaders each get their
// own, independently validated, written, and revision-tracked copy, so
// the same Material instance can be used under more than one shader
// safely. Values not declared by a particular shader are ignored.

import { Appearance } from "./appearance.js";

export class Material extends Appearance {
  constructor (values = {}) {
    super();
    this._values = { ...values };
    this._revision = 0;
  }

  // Writes the intersection of this material's named values and the
  // shader's reflected material fields.
  writeFields (block) {
    for (const name in this._values) {
      if (block.hasField(name)) block.set(name, this._values[name]);
    }
  }

  // Sets an arbitrary material value and invalidates every shader's
  // cached GPU copy of this material.
  set (name, value) {
    this._values[name] = value;
    this._markDirty();
  }

  // Returns a named material value, or `defaultValue` when absent.
  get (name, defaultValue = null) {
    return name in this._values ? this._values[name] : defaultValue;
  }

  // Returns a shallow copy so callers cannot replace entries without
  // incrementing the material revision via set().
  getValues () {
    return { ...this._values };
  }

  // Called by every setter - bumps the revision, so every shader this
  // instance is registered with will rewrite its own block before its
  // next bind (see Shader.bindMaterial).
  _markDirty () {
    this._revision += 1;
  }

  getRevision () {
    return this._revision;
  }

  // Delegates to the active shader - see Shader.bindMaterial.
  load (st) {
    st.getShader().bindMaterial(st, this);
  }

  // Delegates to the active shader - see Shader.unbindMaterial.
  unload (st) {
    st.getShader().unbindMaterial(st);
  }
}
