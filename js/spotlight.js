// Luz spot: luz pontual limitada a um cone em torno de spot_direction.
// Um ponto e iluminado se o angulo entre o eixo do cone e o vetor que vai da
// luz ate o ponto (-L) for menor que a abertura. Comparar angulos e o mesmo
// que comparar cossenos (cos e decrescente em [0, pi]), entao guardamos o
// cosseno e o shader so testa dot(-L, dir) >= cos(abertura), sem acos.

import * as gm from "./graphicsmath.js";
import { Light } from "./light.js";

export class SpotLight extends Light {
  constructor (position, target, cutoffDeg, space = "world", values = {}) {
    super(space, values);
    this.setPosition(position[0], position[1], position[2]);
    this.setTarget(target[0], target[1], target[2]);
    this.setCutoff(cutoffDeg);
  }

  setPosition (x, y, z) {
    this.set("light_position", gm.vec4(x, y, z, 1.0));
  }

  // direcao do cone: da posicao da luz ate o alvo
  setTarget (x, y, z) {
    const p = this.get("light_position");
    this.set("spot_direction", gm.normalize(gm.vec3(x - p[0], y - p[1], z - p[2])));
  }

  // cutoffDeg e o angulo entre o eixo e a borda do cone (meia abertura)
  setCutoff (cutoffDeg) {
    this.set("spot_cos_cutoff", Math.cos(gm.radians(cutoffDeg)));
  }

  // Chamado pelo framework para levar a luz ao espaco de iluminacao do
  // shader. Posicao e ponto (w=1, sofre translacao); direcao e vetor (w=0,
  // so gira), e e normalizada de novo caso a matriz tenha escala.
  map (matrix) {
    const d = this.get("spot_direction");
    const dir = gm.mat4MulVec4(matrix, gm.vec4(d[0], d[1], d[2], 0.0));
    return {
      light_position: gm.mat4MulVec4(matrix, this.get("light_position")),
      spot_direction: gm.normalize(gm.vec3(dir[0], dir[1], dir[2])),
    };
  }
}
