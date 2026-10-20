// Base interface for cameras: projection matrix, view matrix, and
// uniform upload. Defaults return identity; Camera2D/Camera3D override
// them. Ported from python/camera.py.

import * as gm from "./graphicsmath.js";

export class Camera {
  // Projection matrix mapping camera space to clip space, for the given
  // canvas size [width, height] in pixels. Recompute every frame -
  // depends on canvasSize.
  getProjMatrix (canvasSize) {
    return gm.mat4(1);
  }

  // View matrix mapping world space into camera (eye) space. Recomputed
  // every frame since subclasses may derive it from interactive state.
  getViewMatrix () {
    return gm.mat4(1);
  }

  // Pushes this camera's per-frame uniforms onto State's per-field value
  // stacks. Base default pushes nothing; pair any override with a
  // matching unload.
  load (st) {}

  // Pops whatever load pushed. Pairs with load.
  unload (st) {}
}
