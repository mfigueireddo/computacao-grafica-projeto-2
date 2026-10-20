// Vector/matrix helpers for 3D graphics, ported from python/graphics_math.py.
//
// Mat4 is a flat Float32Array(16) in WGSL/WebGPU column-major layout
// (index = col*4 + row) - this is the native layout for mat4x4<f32> uniform
// data, so mat4Bytes is just a copy, not a transpose (unlike the python
// version, which stores row-major and transposes only when packing).
// Composition is right-multiply (translate/scale/rotate apply "before"
// whatever was already accumulated), matching GLM/python's convention.
// Vec2/Vec3/Vec4 are plain Float32Array(2|3|4). All matrices use
// WebGPU's 0..1 NDC depth range, not OpenGL's -1..1.

export function vec2 (x = 0.0, y = undefined) {
  if (y === undefined) y = x;
  return new Float32Array([x, y]);
}

export function vec3 (x = 0.0, y = undefined, z = undefined) {
  if (y === undefined && z === undefined) { y = x; z = x; }
  else if (y === undefined || z === undefined) {
    throw new TypeError("vec3 needs either one value or all of x, y, and z");
  }
  return new Float32Array([x, y, z]);
}

export function vec4 (x = 0.0, y = undefined, z = undefined, w = undefined) {
  if (y === undefined && z === undefined && w === undefined) { y = x; z = x; w = x; }
  else if (y === undefined || z === undefined || w === undefined) {
    throw new TypeError("vec4 needs either one value or all of x, y, z, and w");
  }
  return new Float32Array([x, y, z, w]);
}

export function mat4 (diagonal = 1.0) {
  const m = new Float32Array(16);
  m[0] = diagonal; m[5] = diagonal; m[10] = diagonal; m[15] = diagonal;
  return m;
}

export function mat4FromColumns (c0, c1, c2, c3) {
  const m = new Float32Array(16);
  m.set(_vector(c0, 4, "c0"), 0);
  m.set(_vector(c1, 4, "c1"), 4);
  m.set(_vector(c2, 4, "c2"), 8);
  m.set(_vector(c3, 4, "c3"), 12);
  return m;
}

export function radians (degrees) {
  return degrees * Math.PI / 180.0;
}

export function dot (a, b) {
  let s = 0.0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function cross (a, b) {
  return vec3(
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  );
}

export function length (v) {
  return Math.sqrt(dot(v, v));
}

export function distance (a, b) {
  const d = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) d[i] = a[i] - b[i];
  return length(d);
}

export function normalize (v) {
  const m = length(v);
  if (m === 0.0) throw new Error("cannot normalize a zero-length vector");
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / m;
  return out;
}

// Row/column accessors for the column-major flat layout.
function _at (m, row, col) { return m[col * 4 + row]; }
function _set (m, row, col, value) { m[col * 4 + row] = value; }

export function transpose (matrix) {
  _mat4(matrix, "matrix");
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) _set(out, col, row, _at(matrix, row, col));
  }
  return out;
}

export function inverse (matrix) {
  _mat4(matrix, "matrix");
  const m = matrix;
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (det === 0.0) throw new Error("matrix is singular and cannot be inverted");
  const invDet = 1.0 / det;

  const out = new Float32Array(16);
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * invDet;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * invDet;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * invDet;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * invDet;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * invDet;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * invDet;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * invDet;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * invDet;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;
  return out;
}

export function translation (offset) {
  const [x, y, z] = _components(offset, 3, "offset");
  const m = mat4();
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}

export function scaling (factors) {
  const [x, y, z] = _components(factors, 3, "factors");
  const m = mat4();
  m[0] = x; m[5] = y; m[10] = z;
  return m;
}

export function rotation (angle, axis) {
  const [x, y, z] = normalize(_vector(axis, 3, "axis"));
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1.0 - c;
  const out = new Float32Array(16);
  _set(out, 0, 0, t * x * x + c);   _set(out, 0, 1, t * x * y - s * z); _set(out, 0, 2, t * x * z + s * y); _set(out, 0, 3, 0.0);
  _set(out, 1, 0, t * x * y + s * z); _set(out, 1, 1, t * y * y + c);   _set(out, 1, 2, t * y * z - s * x); _set(out, 1, 3, 0.0);
  _set(out, 2, 0, t * x * z - s * y); _set(out, 2, 1, t * y * z + s * x); _set(out, 2, 2, t * z * z + c);   _set(out, 2, 3, 0.0);
  _set(out, 3, 0, 0.0);              _set(out, 3, 1, 0.0);              _set(out, 3, 2, 0.0);              _set(out, 3, 3, 1.0);
  return out;
}

function _multiply (a, b) {
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      let s = 0.0;
      for (let k = 0; k < 4; k++) s += _at(a, row, k) * _at(b, k, col);
      _set(out, row, col, s);
    }
  }
  return out;
}

export function multiply (a, b) {
  return _product(a, b);
}

// Matrix-vector product treating `v` (length 4) as a column vector.
export function mat4MulVec4 (matrix, v) {
  _mat4(matrix, "matrix");
  const out = new Float32Array(4);
  for (let row = 0; row < 4; row++) {
    let s = 0.0;
    for (let col = 0; col < 4; col++) s += _at(matrix, row, col) * v[col];
    out[row] = s;
  }
  return out;
}

export function translate (matrix, offset) {
  return _product(matrix, translation(offset));
}

export function scale (matrix, factors) {
  return _product(matrix, scaling(factors));
}

export function rotate (matrix, angle, axis) {
  return _product(matrix, rotation(angle, axis));
}

export function lookAt (eye, center, up) {
  const eyeV = _vector(eye, 3, "eye");
  const centerV = _vector(center, 3, "center");
  const forward = normalize(_sub(centerV, eyeV));
  const side = normalize(cross(forward, _vector(up, 3, "up")));
  const correctedUp = cross(side, forward);

  const out = mat4();
  for (let col = 0; col < 3; col++) {
    _set(out, 0, col, side[col]);
    _set(out, 1, col, correctedUp[col]);
    _set(out, 2, col, -forward[col]);
  }
  _set(out, 0, 3, -dot(side, eyeV));
  _set(out, 1, 3, -dot(correctedUp, eyeV));
  _set(out, 2, 3, dot(forward, eyeV));
  return out;
}

export function perspective (fovy, aspect, near, far) {
  if (!(fovy > 0.0 && fovy < Math.PI)) throw new Error(`fovy must be between 0 and pi radians, got ${fovy}`);
  if (!(aspect > 0.0)) throw new Error(`aspect must be > 0, got ${aspect}`);
  _validatePerspectivePlanes(near, far);

  const focalLength = 1.0 / Math.tan(fovy / 2.0);
  const out = new Float32Array(16);
  _set(out, 0, 0, focalLength / aspect);
  _set(out, 1, 1, focalLength);
  _set(out, 2, 2, far / (near - far));
  _set(out, 2, 3, far * near / (near - far));
  _set(out, 3, 2, -1.0);
  return out;
}

export function ortho (left, right, bottom, top, near = -1, far = 1) {
  if (left === right) throw new Error("left and right must differ");
  if (bottom === top) throw new Error("bottom and top must differ");
  if (near === far) throw new Error("near and far must differ");

  const out = mat4();
  _set(out, 0, 0, 2.0 / (right - left));
  _set(out, 1, 1, 2.0 / (top - bottom));
  _set(out, 2, 2, 1.0 / (near - far));
  _set(out, 0, 3, -(right + left) / (right - left));
  _set(out, 1, 3, -(top + bottom) / (top - bottom));
  _set(out, 2, 3, near / (near - far));
  return out;
}

export function mat4Bytes (matrix) {
  _mat4(matrix, "matrix");
  return new Float32Array(matrix);
}

function _sub (a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

function _vector (value, size, name) {
  const result = value instanceof Float32Array ? value : new Float32Array(value);
  if (result.length !== size) throw new Error(`${name} must have length ${size}, got ${result.length}`);
  return result;
}

function _components (value, size, name) {
  return Array.from(_vector(value, size, name));
}

function _mat4 (value, name) {
  if (!(value instanceof Float32Array) || value.length !== 16) {
    throw new Error(`${name} must be a Float32Array of length 16`);
  }
  return value;
}

function _product (left, right) {
  _mat4(left, "left"); _mat4(right, "right");
  return _multiply(left, right);
}

function _validatePerspectivePlanes (near, far) {
  if (!(near > 0.0)) throw new Error(`near must be > 0, got ${near}`);
  if (!(far > near)) throw new Error(`far must be greater than near, got near=${near}, far=${far}`);
}
