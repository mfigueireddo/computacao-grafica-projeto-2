// Accumulates an affine transform as a single 4x4 matrix; translate/
// scale/rotate/multMatrix each right-multiply the current matrix. On
// load, composes this matrix onto the state's matrix stack; unload pops
// it back off. Ported from python/transform.py.

import * as gm from "./graphicsmath.js";

export class Transform {
  // Starts with the identity matrix - no transform applied.
  constructor () {
    this.mat = gm.mat4(1.0);
  }

  // Resets the accumulated matrix to identity, discarding any prior
  // translate/scale/rotate/multMatrix calls.
  loadIdentity () {
    this.mat = gm.mat4(1.0);
  }

  // Right-multiplies the accumulated matrix by `mat`, so `mat` takes
  // effect before whatever was already accumulated.
  multMatrix (mat) {
    this.mat = gm.multiply(this.mat, mat);
  }

  // Right-multiplies the accumulated matrix by a translation of
  // (x, y, z), applied before whatever was already accumulated.
  translate (x, y, z) {
    this.mat = gm.translate(this.mat, gm.vec3(x, y, z));
  }

  // Right-multiplies the accumulated matrix by a scale of (x, y, z),
  // applied before whatever was already accumulated.
  scale (x, y, z) {
    this.mat = gm.scale(this.mat, gm.vec3(x, y, z));
  }

  // Rotate by `angle` degrees around axis (x,y,z).
  rotate (angle, x, y, z) {
    this.mat = gm.rotate(this.mat, gm.radians(angle), gm.vec3(x, y, z));
  }

  getMatrix () {
    return this.mat;
  }

  // Pushes this transform's matrix onto State's matrix stack, composed
  // with the enclosing current matrix. Pair with unload.
  load (st) {
    st.pushMatrix(this.mat);
  }

  // Pops the matrix pushed by load, restoring the enclosing current matrix.
  unload (st) {
    st.popMatrix();
  }
}
