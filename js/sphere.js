// Unit sphere centered at the origin, parameterized over a Grid of
// nstack latitude bands by nslice longitude slices. Vertex buffers: slot
// 0=coord, 1=normal (reuses coordVbo, since position == normal on a unit
// sphere), 2=tangent, 3=texcoord. Ported from python/sphere.py.

import { Shape } from "./shape.js";
import { Grid } from "./grid.js";
import { createBufferWithData } from "./gpuutil.js";

export class Sphere extends Shape {
  constructor (device, nstack = 64, nslice = 64) {
    super();
    const grid = new Grid(nstack, nslice);
    this.nind = grid.indexCount();
    const coord = new Float32Array(3 * grid.vertexCount());
    const tangent = new Float32Array(3 * grid.vertexCount());
    // the parameterization comes from getCoords (v growing from south to
    // north pole); what goes to the GPU is getTexcoords, with t growing downward
    const param = grid.getCoords();
    const texcoord = grid.getTexcoords();
    let nc = 0;
    for (let i = 0; i < 2 * grid.vertexCount(); i += 2) {
      const theta = param[i + 0] * 2 * Math.PI;
      const phi = param[i + 1] * Math.PI;
      coord[nc + 0] = Math.sin(theta) * Math.sin(Math.PI - phi);
      coord[nc + 1] = Math.cos(Math.PI - phi);
      coord[nc + 2] = Math.cos(theta) * Math.sin(Math.PI - phi);
      tangent[nc + 0] = Math.cos(theta);
      tangent[nc + 1] = 0;
      tangent[nc + 2] = -Math.sin(theta);
      nc += 3;
    }

    this.coordVbo = createBufferWithData(device, coord, GPUBufferUsage.VERTEX);
    // the same mesh (coord) serves as normal, since this is a unit sphere centered at the origin
    this.tangentVbo = createBufferWithData(device, tangent, GPUBufferUsage.VERTEX);
    this.texcoordVbo = createBufferWithData(device, texcoord, GPUBufferUsage.VERTEX);
    this.ibo = createBufferWithData(device, grid.getIndices(), GPUBufferUsage.INDEX);
  }

  // Reuses coordVbo for the normal slot before the drawIndexed call.
  draw (st) {
    const firstInstance = st.getShader().commitMatrix(st);
    st.renderPass.setVertexBuffer(0, this.coordVbo);
    st.renderPass.setVertexBuffer(1, this.coordVbo); // normal == coord (unit sphere)
    st.renderPass.setVertexBuffer(2, this.tangentVbo);
    st.renderPass.setVertexBuffer(3, this.texcoordVbo);
    st.renderPass.setIndexBuffer(this.ibo, "uint32");
    st.renderPass.drawIndexed(this.nind, 1, 0, 0, firstInstance);
  }
}
