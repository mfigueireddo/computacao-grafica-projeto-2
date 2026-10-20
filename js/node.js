// A scene graph node: holds an optional Pipeline/Transform, this node's
// own Appearances and Shapes, and child Nodes. Pipeline, Transform, and
// Appearances (materials, texture sets) are all inherited by descendants
// that don't set their own - only Shapes are strictly local to a Node.
// Ported from python/node.py.

import * as gm from "./graphicsmath.js";

export class Node {
  // Builds a node from already-constructed parts; `nodes` (if given) are
  // attached via addNode.
  constructor ({ pipeline = null, trf = null, apps = [], shps = [], nodes = [] } = {}) {
    this.parent = null;
    this.pipeline = pipeline;
    this.trf = trf;
    this.apps = apps;
    this.shps = shps;
    this.nodes = [];
    for (const n of nodes) this.addNode(n);
  }

  // Sets the Pipeline this node loads on render; descendants that don't
  // set their own inherit it from an ancestor.
  setPipeline (pipeline) {
    this.pipeline = pipeline;
  }

  // Returns the Pipeline set directly on this node, or null if it only
  // inherits one from an ancestor.
  getPipeline () {
    return this.pipeline;
  }

  // Sets the local Transform composed into this node's model matrix, or
  // clears it (null) back to identity; descendants without their own
  // Transform get identity here, not an ancestor's.
  setTransform (trf) {
    this.trf = trf;
  }

  // Appends `app` to this node's appearance list. Loaded in order
  // (unloaded in reverse) before this node's Shapes are drawn, so later
  // appearances can override earlier ones.
  addAppearance (app) {
    this.apps.push(app);
  }

  // Appends `shp` to this node's shape list; drawn in order under
  // whichever Pipeline is active when this node renders.
  addShape (shp) {
    this.shps.push(shp);
  }

  // Appends `node` as a child and sets this node as its parent. Throws if
  // `node` already has a parent (use removeNode on the current parent
  // first) or if `node` is `self` or one of `self`'s own ancestors (which
  // would make the tree an infinite loop for render()/getModelMatrix()
  // to walk).
  addNode (node) {
    if (node.parent !== null) {
      throw new Error("Node already has a parent - remove it from its current parent first");
    }
    let ancestor = this;
    while (ancestor !== null) {
      if (ancestor === node) throw new Error("Node is self or an ancestor of self - would create a cycle");
      ancestor = ancestor.parent;
    }
    this.nodes.push(node);
    node.setParent(this);
  }

  // Detaches `node` from this Node's children and clears its parent
  // pointer, so it (or a subtree containing it) can be added elsewhere.
  // Throws if `node` isn't currently a child of this Node.
  removeNode (node) {
    const i = this.nodes.indexOf(node);
    if (i === -1) throw new Error("node is not a child of this Node");
    this.nodes.splice(i, 1);
    node.parent = null;
  }

  // Sets this node's parent pointer, used by getModelMatrix to walk up
  // the ancestor chain. Called automatically by addNode, not meant to be
  // called directly.
  setParent (parent) {
    this.parent = parent;
  }

  // Returns this node's parent, or null if it's the root (or not yet
  // attached to a tree).
  getParent () {
    return this.parent;
  }

  // Returns this node's own local transform matrix, or the identity if no
  // Transform is set.
  getMatrix () {
    return this.trf ? this.trf.getMatrix() : gm.mat4(1.0);
  }

  // Returns the accumulated model matrix: this node's local matrix
  // composed with every ancestor's, up to the root.
  getModelMatrix () {
    let mat = this.getMatrix();
    let node = this.getParent();
    while (node) {
      mat = gm.multiply(node.getMatrix(), mat);
      node = node.getParent();
    }
    return mat;
  }

  // Recursively renders this node's subtree: loads its own
  // Pipeline/Transform/Appearances, draws its own Shapes, renders every
  // child Node, then unloads everything in reverse order.
  render (st) {
    // load
    if (this.pipeline) this.pipeline.load(st);
    if (this.trf) this.trf.load(st);
    for (const app of this.apps) app.load(st);
    // draw
    if (this.shps.length > 0) {
      // whichever Pipeline is current is already bound on the render pass:
      // Pipeline.load binds it on the way in, and a nested Pipeline's
      // unload rebinds the enclosing one on the way out
      st.loadMatrices();
      for (const shp of this.shps) shp.draw(st);
      st.unloadMatrices();
    }
    for (const node of this.nodes) node.render(st);
    // unload in reverse order
    for (let i = this.apps.length - 1; i >= 0; i--) this.apps[i].unload(st);
    if (this.trf) this.trf.unload(st);
    if (this.pipeline) this.pipeline.unload(st);
  }
}
