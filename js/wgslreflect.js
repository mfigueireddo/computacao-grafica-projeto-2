// WGSL reflection, ported from python/wgslreflect.py: regex-based (not a
// full parser), extracts bind group layouts, uniform/storage-array field
// schemas, and vertex input locations from WGSL source text. Missing-name
// queries throw; some callers catch for silent-skip semantics, others let
// it surface as a validation error.

// Subset of WGSL types this project supports (see uniformbuffer.js _ALIGN_SIZE).
const NORMALIZABLE = new Set(["i32", "f32", "vec2", "vec3", "vec4", "mat4x4"]);

const STRUCT_RE = /struct\s+(\w+)\s*\{([^}]*)\}/gs;

// WGSL comments, stripped before any parsing: a trailing "// ..." on a struct
// field line would otherwise be glued onto the *next* field's name (fields are
// split on ","), silently making hasField() miss it - and a miss means the
// field is never written, so it reads as zero on the GPU.
const COMMENT_RE = /\/\/[^\n]*|\/\*[\s\S]*?\*\//g;

// Removes line and block comments, keeping newlines so anything reported by
// line number still lines up.
function stripComments (code) {
  return code.replace(COMMENT_RE, (m) => "\n".repeat((m.match(/\n/g) || []).length));
}
const ATTR_PREFIX_RE = /@\w+(?:\([^)]*\))?\s*/g;
// group 3 captures whatever's inside var<...> (e.g. "uniform" or
// "storage, read") - undefined for a bare `var` (textures/samplers).
const BINDING_RE = /@group\(\s*(\d+)\s*\)\s*@binding\(\s*(\d+)\s*\)\s*var(?:<([^>]*)>)?\s+(\w+)\s*:\s*([^;]+);/g;
const STORAGE_ARRAY_RE = /^array<\s*(\w+)\s*>$/;
const VERTEX_MAIN_START_RE = /fn\s+vs_main\s*\(/;
const VERTEX_PARAM_RE = /@location\(\s*(\d+)\s*\)\s*(\w+)\s*:\s*([^,]+)/g;

const TEXTURE_VIEW_DIM = {
  texture_1d: "1d",
  texture_2d: "2d",
  texture_2d_array: "2d-array",
  texture_cube: "cube",
  texture_cube_array: "cube-array",
  texture_3d: "3d",
  texture_depth_2d: "2d",
  texture_depth_2d_array: "2d-array",
  texture_depth_cube: "cube",
  texture_depth_cube_array: "cube-array",
};

// Returns vs_main's parameter list text, or null if there's no vs_main.
// Tracks paren depth manually since the params themselves contain parens
// (@location(0)).
function extractVsMainParams (code) {
  const m = VERTEX_MAIN_START_RE.exec(code);
  if (!m) return null;
  let depth = 1;
  let i = m.index + m[0].length;
  const start = i;
  while (i < code.length && depth > 0) {
    if (code[i] === "(") depth++;
    else if (code[i] === ")") depth--;
    i++;
  }
  return code.slice(start, i - 1);
}

// Resolves one raw field type to either a primitive name (one of
// NORMALIZABLE) or, for a reference to another struct declared in this
// same file, a nested schema - an array of [name, resolvedType] pairs,
// each resolved the same way. `stack` is the chain of struct names being
// resolved, to throw instead of recursing forever on a cyclic struct
// reference.
function normalizeType (wgslType, structs, stack = []) {
  const base = wgslType.split("<")[0].trim();
  if (NORMALIZABLE.has(base)) return base;
  if (base in structs) {
    if (stack.includes(base)) {
      throw new Error(`Recursive struct definition: ${[...stack, base].join(" -> ")}`);
    }
    return structs[base].map(([fname, ftype]) => [fname, normalizeType(ftype, structs, [...stack, base])]);
  }
  throw new Error(
    `Unsupported WGSL type in uniform block: '${wgslType}' ` +
    `(supported: ${[...NORMALIZABLE].sort()}, or another struct declared in this file)`
  );
}

function parseStructs (code) {
  const structs = {};
  for (const m of code.matchAll(STRUCT_RE)) {
    const name = m[1];
    const fields = [];
    for (let part of m[2].split(",")) {
      part = part.replace(ATTR_PREFIX_RE, "").trim();
      if (!part) continue;
      const colon = part.indexOf(":");
      if (colon === -1) continue;
      const fname = part.slice(0, colon).trim();
      const ftype = part.slice(colon + 1).trim();
      if (fname && ftype) fields.push([fname, ftype]);
    }
    structs[name] = fields;
  }
  return structs;
}

function textureEntry (binding, wgslType) {
  const base = wgslType.split("<")[0].trim();
  const sampleType = base.startsWith("texture_depth") ? "depth" : "float";
  const viewDimension = TEXTURE_VIEW_DIM[base] ?? "2d";
  return { binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType, viewDimension } };
}

function samplerEntry (binding, wgslType) {
  const kind = wgslType === "sampler_comparison" ? "comparison" : "filtering";
  return { binding, visibility: GPUShaderStage.FRAGMENT, sampler: { type: kind } };
}

function uniformEntry (binding) {
  return { binding, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } };
}

function storageArrayEntry (binding) {
  return { binding, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } };
}

export class ShaderReflection {
  // Parses WGSL source `code` once, caching the result for the query
  // methods below.
  constructor (code) {
    this._data = this._parse(stripComments(code));
  }

  _parse (code) {
    const structs = parseStructs(code);
    const groups = new Map();
    const fields = new Map();
    const textures = new Map();
    const samplers = new Map();
    const uniformVars = new Map();

    const group = (index) => {
      if (!groups.has(index)) groups.set(index, { entries: [], uniformFields: null, storageArrayFields: null });
      return groups.get(index);
    };

    const registerFields = (idx, varname, rawFields) => {
      const normalized = rawFields.map(([fname, ftype]) => [fname, normalizeType(ftype, structs)]);
      uniformVars.set(varname, idx);
      for (const [fname] of normalized) {
        if (fields.has(fname) && fields.get(fname) !== idx) {
          throw new Error(
            `Field '${fname}' declared in more than one group (${fields.get(fname)} and ${idx}) - ` +
            "field names must be unique across a shader's uniform blocks"
          );
        }
        fields.set(fname, idx);
      }
      return normalized;
    };

    for (const m of code.matchAll(BINDING_RE)) {
      const idx = parseInt(m[1], 10);
      const binding = parseInt(m[2], 10);
      const addrSpace = (m[3] ?? "").trim();
      const varname = m[4];
      const typeExpr = m[5].trim();
      const g = group(idx);

      if (typeExpr.startsWith("texture")) {
        g.entries.push(textureEntry(binding, typeExpr));
        textures.set(varname, [idx, binding]);
        continue;
      }

      if (typeExpr === "sampler" || typeExpr === "sampler_comparison") {
        g.entries.push(samplerEntry(binding, typeExpr));
        samplers.set(varname, [idx, binding]);
        continue;
      }

      const arrayMatch = STORAGE_ARRAY_RE.exec(typeExpr);
      if (addrSpace.startsWith("storage") && arrayMatch) {
        // var<storage, read> varname: array<StructType>
        const structName = arrayMatch[1];
        const rawFields = structs[structName];
        if (rawFields === undefined) {
          throw new Error(`Storage array '${varname}' (group ${idx}) references undeclared struct: '${structName}'`);
        }
        g.entries.push(storageArrayEntry(binding));
        g.storageArrayFields = registerFields(idx, varname, rawFields);
        continue;
      }

      // var<uniform> varname: StructType
      const rawFields = structs[typeExpr];
      if (rawFields === undefined) {
        throw new Error(`Uniform '${varname}' (group ${idx}) references undeclared struct: '${typeExpr}'`);
      }
      g.entries.push(uniformEntry(binding));
      g.uniformFields = registerFields(idx, varname, rawFields);
    }

    const indices = [...groups.keys()].sort((a, b) => a - b);
    if (indices.length > 0 && indices.some((v, i) => v !== i)) {
      throw new Error(
        `Bind group indices must be contiguous starting at 0; found: ${indices}. ` +
        "WebGPU doesn't allow 'skipping' a group index."
      );
    }

    // name -> location; format/size aren't reflected at all - WebGPU lets an
    // application's buffer supply a smaller vector format than vs_main
    // declares, so only the application (via Shader.setVertexBuffers)
    // knows the actual packing, never reflection.
    const vertexInputs = new Map();
    const vsMainParams = extractVsMainParams(code);
    if (vsMainParams !== null) {
      for (const pm of vsMainParams.matchAll(VERTEX_PARAM_RE)) {
        vertexInputs.set(pm[2], parseInt(pm[1], 10));
      }
    }

    const groupVarNames = new Map();
    for (const [varname, idx] of uniformVars) groupVarNames.set(idx, varname);

    return { groups, fields, textures, samplers, vertexInputs, uniformVars, groupVarNames };
  }

  // --- bind group queries, used by Shader to build GPUBindGroupLayouts ---

  // Every declared @group index, sorted ascending; contiguous from 0.
  groupIndices () {
    return [...this._data.groups.keys()].sort((a, b) => a - b);
  }

  // GPUBindGroupLayoutEntry objects for `group`, in WGSL declaration order.
  layoutEntries (group) {
    return this._data.groups.get(group).entries;
  }

  // [fieldName, wgslType] pairs for the uniform struct bound in `group`, in
  // declaration order. wgslType is a primitive type name, or - for a field
  // whose type is itself a struct declared in this file - a nested array of
  // the same shape (see normalizeType); uniformbuffer.js's layout code is
  // what flattens that into dotted leaf fields.
  uniformFields (group) {
    return this._data.groups.get(group).uniformFields;
  }

  // Same as uniformFields but for the element struct of the
  // `var<storage, read> name: array<StructType>` bound in `group`; null if
  // `group` isn't a storage array.
  storageArrayFields (group) {
    return this._data.groups.get(group).storageArrayFields;
  }

  // Bind group index of the uniform variable literally named `varname`
  // (e.g. "material", "global", "matrix"); throws if undeclared.
  uniformVarGroup (varname) {
    if (!this._data.uniformVars.has(varname)) throw new Error(`KeyError: ${varname}`);
    return this._data.uniformVars.get(varname);
  }

  // Uniform variable name bound in `group` - inverse of uniformVarGroup;
  // throws if `group` has no uniform var (e.g. a texture/sampler-only group).
  groupVarName (group) {
    if (!this._data.groupVarNames.has(group)) throw new Error(`KeyError: ${group}`);
    return this._data.groupVarNames.get(group);
  }

  // --- named-field queries ---

  // Bind group index of the uniform field `name`; throws if undeclared.
  // `name` is the top-level field name only.
  fieldGroup (name) {
    if (!this._data.fields.has(name)) throw new Error(`KeyError: ${name}`);
    return this._data.fields.get(name);
  }

  // [group, binding] for the texture variable `name`.
  textureBinding (name) {
    if (!this._data.textures.has(name)) throw new Error(`KeyError: ${name}`);
    return this._data.textures.get(name);
  }

  // [group, binding] for the sampler variable `name`.
  samplerBinding (name) {
    if (!this._data.samplers.has(name)) throw new Error(`KeyError: ${name}`);
    return this._data.samplers.get(name);
  }

  // --- vertex input queries, used by Shader to resolve setVertexBuffers' varName ---

  // @location vs_main declares for input `name`; throws if not found.
  vertexLocation (name) {
    if (!this._data.vertexInputs.has(name)) throw new Error(`KeyError: ${name}`);
    return this._data.vertexInputs.get(name);
  }

  // Every @location this shader's vs_main declares, for Shader to verify coverage.
  vertexLocations () {
    return new Set(this._data.vertexInputs.values());
  }
}
