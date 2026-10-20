// Base for Node-loaded state (materials, texture sets, ...). Ported from
// python/appearance.py.
export class Appearance {
  load (st) {
    throw new Error("not implemented");
  }

  unload (st) {}
}
