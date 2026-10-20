// Emulates GLSL's gl_ClipDistance (no WGSL equivalent) via a per-fragment
// discard in the shader. Sets "clip_plane" (ax+by+cz+d=0) and
// "clip_plane_color" (cut cross-section color) directly on `shader` - a
// "global" group field, throwing if `shader`'s "global" struct doesn't
// declare them (see Shader.setValue). Ported from python/clipplane.py.

import * as gm from "./graphicsmath.js";

export class ClipPlane {
  constructor (shader, a, b, c, d) {
    this.shader = shader;
    this.setPlane(a, b, c, d);
    this.setColor(0.5, 0.5, 0.5);
  }

  // Sets the plane equation ax+by+cz+d=0, stored as gm.vec4(a,b,c,d).
  setPlane (a, b, c, d) {
    this.shader.setValue("clip_plane", gm.vec4(a, b, c, d));
  }

  setColor (r, g, b) {
    this.shader.setValue("clip_plane_color", gm.vec4(r, g, b, 1.0));
  }
}
