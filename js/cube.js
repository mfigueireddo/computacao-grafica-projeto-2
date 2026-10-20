// Axis-aligned unit cube spanning x,z in [-0.5,0.5], y in [0,1]. Vertex
// buffers: slot 0=coord, 1=normal, 2=tangent, 3=texcoord, plus a uint32
// index buffer (36 indices). Ported from python/cube.py.

import { Shape } from "./shape.js";
import { createBufferWithData } from "./gpuutil.js";

export class Cube extends Shape {
  constructor (device) {
    super();
    const coords = new Float32Array([
      // back face: counter clockwise
      -0.5, 0.0, -0.5,
      -0.5, 1.0, -0.5,
      0.5, 1.0, -0.5,
      0.5, 0.0, -0.5,
      // front face: counter clockwise
      -0.5, 0.0, 0.5,
      0.5, 0.0, 0.5,
      0.5, 1.0, 0.5,
      -0.5, 1.0, 0.5,
      // left face: counter clockwise
      -0.5, 0.0, -0.5,
      -0.5, 0.0, 0.5,
      -0.5, 1.0, 0.5,
      -0.5, 1.0, -0.5,
      // right face: counter clockwise
      0.5, 0.0, -0.5,
      0.5, 1.0, -0.5,
      0.5, 1.0, 0.5,
      0.5, 0.0, 0.5,
      // bottom face: counter clockwise
      -0.5, 0.0, -0.5,
      0.5, 0.0, -0.5,
      0.5, 0.0, 0.5,
      -0.5, 0.0, 0.5,
      // top face: counter clockwise
      -0.5, 1.0, -0.5,
      -0.5, 1.0, 0.5,
      0.5, 1.0, 0.5,
      0.5, 1.0, -0.5,
    ]);
    const normals = new Float32Array([
      0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
      -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
      1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0,
      0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    ]);
    // t grows downward: the bottom pair of vertices of each face gets t = 1
    const texcoords = new Float32Array([
      0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    ]);
    const tangents = new Float32Array([
      -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
      1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
      0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0,
      -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
      1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ]);
    const index = new Uint32Array([
      0, 1, 2, 0, 2, 3,
      4, 5, 6, 4, 6, 7,
      8, 9, 10, 8, 10, 11,
      12, 13, 14, 12, 14, 15,
      16, 17, 18, 16, 18, 19,
      20, 21, 22, 20, 22, 23,
    ]);

    this.coordVbo = createBufferWithData(device, coords, GPUBufferUsage.VERTEX);
    this.normalVbo = createBufferWithData(device, normals, GPUBufferUsage.VERTEX);
    this.tangentVbo = createBufferWithData(device, tangents, GPUBufferUsage.VERTEX);
    this.texcoordVbo = createBufferWithData(device, texcoords, GPUBufferUsage.VERTEX);
    this.ibo = createBufferWithData(device, index, GPUBufferUsage.INDEX);
  }

  draw (st) {
    const firstInstance = st.getShader().commitMatrix(st);
    st.renderPass.setVertexBuffer(0, this.coordVbo);
    st.renderPass.setVertexBuffer(1, this.normalVbo);
    st.renderPass.setVertexBuffer(2, this.tangentVbo);
    st.renderPass.setVertexBuffer(3, this.texcoordVbo);
    st.renderPass.setIndexBuffer(this.ibo, "uint32");
    st.renderPass.drawIndexed(36, 1, 0, 0, firstInstance);
  }
}
