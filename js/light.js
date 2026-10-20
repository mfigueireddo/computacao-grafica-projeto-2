// Base for shader-defined lights. Ported from python/light.py.
//
// Light stores arbitrary named values and handles the source-space to
// shader-space plumbing. It deliberately does not know which values are
// geometric: every concrete light implements map(matrix) from scratch.

import * as gm from "./graphicsmath.js";

export class Light {
  constructor (space = "world", values = {}) {
    if (!["world", "camera", "local"].includes(space)) {
      throw new Error("Light space must be 'world', 'camera', or 'local'");
    }
    this.space = space;
    this.reference = null;
    this._values = {
      light_ambient: gm.vec3(0.2, 0.2, 0.2),
      light_diffuse: gm.vec3(0.8, 0.8, 0.8),
      light_specular: gm.vec3(1.0, 1.0, 1.0),
      ...values,
    };
  }

  // Sets an arbitrary shader light parameter.
  set (name, value) {
    this._values[name] = value;
  }

  get (name, defaultValue = null) {
    return name in this._values ? this._values[name] : defaultValue;
  }

  // Returns a shallow copy of the stored source values.
  getValues () {
    return { ...this._values };
  }

  setAmbient (r, g, b) {
    this.set("light_ambient", gm.vec3(r, g, b));
  }

  setDiffuse (r, g, b) {
    this.set("light_diffuse", gm.vec3(r, g, b));
  }

  setSpecular (r, g, b) {
    this.set("light_specular", gm.vec3(r, g, b));
  }

  // Sets the node whose local space contains this light's values.
  // References are meaningful only for a light explicitly constructed
  // with space="local"; world- and camera-space lights reject this
  // operation.
  setReference (reference) {
    if (this.space !== "local") throw new Error("A reference node can only be assigned to a local-space light");
    this.reference = reference;
  }

  getReference () {
    return this.reference;
  }

  // Returns this light's values mapped by `matrix`. Concrete lights must
  // implement all of their own mapping semantics and must not mutate the
  // stored source values.
  map (matrix) {
    throw new Error("not implemented");
  }

  // Returns the matrix from this light's source space to the shader's
  // world- or camera-space lighting coordinates.
  getMappingMatrix (st, lightingSpace) {
    if (!["world", "camera"].includes(lightingSpace)) {
      throw new Error("Shader lighting space must be 'world' or 'camera'");
    }

    if (this.space === "local") {
      if (this.reference === null) throw new Error("Local-space light has no reference node");
      const model = this.reference.getModelMatrix();
      if (lightingSpace === "world") return model;
      return gm.multiply(st.getViewMatrix(), model);
    }

    if (this.space === lightingSpace) return gm.mat4(1.0);
    if (this.space === "world") return st.getViewMatrix();
    return st.getInverseViewMatrix();
  }

  // Builds the source-to-lighting-space matrix, delegates geometric
  // semantics to map(), and writes fields declared by the shader.
  writeFields (block, st, lightingSpace) {
    const matrix = this.getMappingMatrix(st, lightingSpace);
    const values = { ...this.getValues(), ...this.map(matrix) };
    for (const name in values) {
      if (block.hasField(name)) block.set(name, values[name]);
    }
  }
}
