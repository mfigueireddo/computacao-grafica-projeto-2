// Projetor de slide (textura projetiva). A ideia e tratar o projetor como
// se fosse uma camera colocada na cena: um ponto do mundo que essa camera
// "ve" cai numa posicao da imagem dela, e e essa posicao que usamos como
// coordenada de textura do slide.
//   View: mundo -> espaco do projetor
//   Proj: espaco do projetor -> coordenadas normalizadas [-1,1] (apos /w)
//   Bias: [-1,1] -> [0,1], que e o intervalo das coordenadas de textura
// tex_matrix = Bias * Proj * View

import * as gm from "./graphicsmath.js";

export class Projector {
  constructor (position, target, fovy = 40, aspect = 1.0) {
    this.position = gm.vec3(position[0], position[1], position[2]);
    this.target = gm.vec3(target[0], target[1], target[2]);
    this.up = gm.vec3(0, 0, -1); // olha para baixo, entao o up nao pode ser (0,1,0)
    this.fovy = fovy;
    this.aspect = aspect;
  }

  getViewMatrix () {
    return gm.lookAt(this.position, this.target, this.up);
  }

  getProjMatrix () {
    return gm.perspective(gm.radians(this.fovy), this.aspect, 0.1, 100.0);
  }

  // A divisao por w (perspectiva) fica para o shader, por pixel.
  getTexMatrix () {
    // s = 0.5x + 0.5 e t = -0.5y + 0.5: o y e invertido porque no WebGPU o t
    // da textura cresce para baixo (linha 0 = topo da imagem). Sem isso o
    // slide aparece de cabeca para baixo.
    const bias = gm.mat4FromColumns(
      [0.5, 0.0, 0.0, 0.0],
      [0.0, -0.5, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.5, 0.5, 0.0, 1.0],
    );
    return gm.multiply(bias, gm.multiply(this.getProjMatrix(), this.getViewMatrix()));
  }
}
