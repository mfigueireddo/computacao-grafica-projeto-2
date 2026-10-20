// Generic indexed nx-by-ny grid mesh generator, shared by Quad and
// Sphere. nx/ny are cell counts, not vertex counts - there are
// (nx+1)x(ny+1) vertices. Carries two parallel arrays: coords, with y
// growing up (the geometry), and texcoords, with t growing down (the
// WebGPU convention, origin at the image's top-left corner). Ported from
// python/grid.py.

export class Grid {
  // Builds the vertex coordinate and index arrays as plain typed arrays
  // (no GPU buffers - callers upload those). Throws if nx/ny isn't
  // positive.
  constructor (nx, ny) {
    if (nx <= 0 || ny <= 0) throw new Error(`Grid needs nx > 0 and ny > 0, got nx=${nx}, ny=${ny}`);
    this.nx = nx;
    this.ny = ny;
    this.coords = new Float32Array(2 * this.vertexCount());
    this.texcoords = new Float32Array(2 * this.vertexCount());
    const dx = 1 / nx;
    const dy = 1 / ny;
    let nc = 0;
    for (let j = 0; j <= ny; j++) {
      for (let i = 0; i <= nx; i++) {
        this.coords[nc + 0] = i * dx;
        this.coords[nc + 1] = j * dy;
        this.texcoords[nc + 0] = i * dx;
        this.texcoords[nc + 1] = 1 - j * dy; // t grows downward
        nc += 2;
      }
    }

    // Flattens grid coordinate (i, j) into a row-major vertex index.
    const findex = (i, j) => j * (nx + 1) + i;

    this.indices = new Uint32Array(this.indexCount());
    let ni = 0;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        this.indices[ni + 0] = findex(i, j);
        this.indices[ni + 1] = findex(i + 1, j);
        this.indices[ni + 2] = findex(i + 1, j + 1);
        this.indices[ni + 3] = findex(i, j);
        this.indices[ni + 4] = findex(i + 1, j + 1);
        this.indices[ni + 5] = findex(i, j + 1);
        ni += 6;
      }
    }
  }

  getNx () { return this.nx; }
  getNy () { return this.ny; }

  // Total vertex count: (nx+1)*(ny+1).
  vertexCount () {
    return (this.nx + 1) * (this.ny + 1);
  }

  // Flat float32 array of interleaved (x,y) coords in [0,1]x[0,1], length 2*vertexCount().
  getCoords () {
    return this.coords;
  }

  // Flat float32 array of interleaved (s,t) coords in [0,1]x[0,1]. Same
  // order as getCoords(), but with t mirrored: t = 1 - y.
  getTexcoords () {
    return this.texcoords;
  }

  // Total index count: 6*nx*ny (2 triangles per cell).
  indexCount () {
    return 6 * this.nx * this.ny;
  }

  // Flat uint32 triangle-list index array, length indexCount().
  getIndices () {
    return this.indices;
  }
}
