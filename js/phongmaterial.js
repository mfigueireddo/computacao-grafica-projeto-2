// Blinn-Phong surface appearance: a diffuse base color, a specular color,
// shininess exponent, and opacity. Ported from python/phongmaterial.py.
// The ambient term isn't a material field - shaders source it straight
// from the "global" group's light.

import * as gm from "./graphicsmath.js";
import { Material } from "./material.js";

export class PhongMaterial extends Material {
  constructor (r, g, b, opacity = 1.0) {
    super({
      base_color: gm.vec3(r, g, b),
      opacity,
      specular_color: gm.vec3(1, 1, 1),
      shininess: 32.0,
    });
  }

  setBaseColor (r, g, b) {
    this.set("base_color", gm.vec3(r, g, b));
  }

  setSpecular (r, g, b) {
    this.set("specular_color", gm.vec3(r, g, b));
  }

  setShininess (shi) {
    this.set("shininess", shi);
  }

  // Only has a visible effect with a shader that actually reads
  // material.opacity in its fragment stage (most don't) *and* a Pipeline
  // configured for blending.
  setOpacity (opacity) {
    this.set("opacity", opacity);
  }
}
