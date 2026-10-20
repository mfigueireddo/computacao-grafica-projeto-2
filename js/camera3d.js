// Perspective or orthographic 3D camera positioned by an eye/center/up
// triad. Optional Arcball adds interactive orbit rotation; optional
// reference Node's inverse model matrix composes into the view so the
// camera follows it. Ported from python/camera3d.py.

import * as gm from "./graphicsmath.js";
import { Camera } from "./camera.js";
import { Arcball } from "./arcball.js";

export class Camera3D extends Camera {
  // Places the eye at (x,y,z), looking at the world origin with +Y up,
  // perspective by default (fovy=45, znear=0.1, zfar=1000); arcball and
  // reference node start unset.
  constructor (x, y, z) {
    super();
    this.ortho = false;
    this.fovy = 45;
    this.znear = 0.1;
    this.zfar = 1000;
    this.center = gm.vec3(0, 0, 0);
    this.eye = gm.vec3(x, y, z);
    this.up = gm.vec3(0, 1, 0);
    this.arcball = null;
    this.reference = null;
  }

  // Sets the vertical field of view, in degrees.
  setAngle (fovy) {
    this.fovy = fovy;
  }

  getAngle () {
    return this.fovy;
  }

  // Sets the near/far clip plane distances used by getProjMatrix, for
  // both the perspective and orthographic cases.
  setZPlanes (znear, zfar) {
    this.znear = znear;
    this.zfar = zfar;
  }

  // Sets the look-at target point, in world space.
  setCenter (x, y, z) {
    this.center = gm.vec3(x, y, z);
  }

  getCenter () {
    return this.center;
  }

  // Sets the camera's eye (viewpoint) position, in world space.
  setEye (x, y, z) {
    this.eye = gm.vec3(x, y, z);
  }

  getEye () {
    return this.eye;
  }

  // Sets the up direction for the look-at view matrix; needn't be
  // normalized or orthogonal to the view direction.
  setUpDir (x, y, z) {
    this.up = gm.vec3(x, y, z);
  }

  // Toggles getProjMatrix between perspective (false, default) and
  // orthographic (true) projection.
  setOrtho (flag) {
    this.ortho = flag;
  }

  // Creates and attaches an Arcball sized to the camera's current
  // eye-to-center distance, so its rotation pivots around the center.
  createArcball () {
    const d = gm.distance(this.eye, this.center);
    this.arcball = new Arcball(d);
    return this.arcball;
  }

  // Returns the attached Arcball, or null if createArcball was never called.
  getArcball () {
    return this.arcball;
  }

  // Attaches the camera to a scene node; the node's model matrix is
  // inverted and composed into the view matrix so the camera follows it.
  setReference (ref) {
    this.reference = ref;
  }

  // Perspective or orthographic projection matrix (per setOrtho) for the
  // given canvasSize. Recomputed every frame.
  getProjMatrix (canvasSize) {
    const [w, h] = canvasSize;
    if (w <= 0 || h <= 0) throw new Error(`getProjMatrix needs a canvasSize with both dimensions > 0, got ${canvasSize}`);
    const ratio = w / h;
    if (!this.ortho) {
      return gm.perspective(gm.radians(this.fovy), ratio, this.znear, this.zfar);
    } else {
      const dist = gm.distance(this.eye, this.center);
      const height = dist * Math.tan(gm.radians(this.fovy) / 2);
      const width = height / h * w;
      return gm.ortho(-width, width, -height, height, this.znear, this.zfar);
    }
  }

  // View matrix transforming world space into camera (eye) space:
  // composes the arcball rotation (if attached), the eye/center/up
  // look-at, then the reference node's inverse transform (if attached),
  // in that order. Recomputed every frame.
  getViewMatrix () {
    let view = gm.mat4(1.0);
    if (this.arcball) view = gm.multiply(view, this.arcball.getMatrix());
    view = gm.multiply(view, gm.lookAt(this.eye, this.center, this.up));
    if (this.reference) view = gm.multiply(view, gm.inverse(this.reference.getModelMatrix()));
    return view;
  }
}
