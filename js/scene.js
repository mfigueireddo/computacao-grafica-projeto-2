// Top-level container: a root Node plus the Engines (simulation/logic
// updated once per frame, e.g. animation or physics) that drive it.
// Ported from python/scene.py.

export class Scene {
  constructor (root) {
    this.root = root;
    this.engines = [];
  }

  getRoot () {
    return this.root;
  }

  addEngine (engine) {
    this.engines.push(engine);
  }

  // Advances every registered Engine by `dt` seconds. Call once per
  // frame, before render.
  update (dt) {
    for (const e of this.engines) e.update(dt);
  }

  // Walks the scene graph from the root, issuing draw calls through
  // `state`. Call once per render pass/camera, after update.
  render (state) {
    this.root.render(state);
  }
}
