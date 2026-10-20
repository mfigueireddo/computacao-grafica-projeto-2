// Projeto 2 (INF1761) - Opcao 1: mesa e objetos
// Teclas: F fog, C plano de corte, L 2a luz, B bump, P projetor
//
// Fluxo geral: main() inicializa o WebGPU, initialize() monta tudo uma vez
// (camera, luz, shader, texturas, materiais e o grafo de cena) e draw() e
// chamado a cada quadro para percorrer o grafo e desenhar.

import * as gm from "./graphicsmath.js";
import { Camera3D } from "./camera3d.js";
import { SpotLight } from "./spotlight.js";
import { Projector } from "./projector.js";
import { ClipPlane } from "./clipplane.js";
import { Shader } from "./shader.js";
import { Pipeline } from "./pipeline.js";
import { PhongMaterial } from "./phongmaterial.js";
import { Texture } from "./texture.js";
import { Sampler } from "./sampler.js";
import { TextureSet } from "./textureset.js";
import { Transform } from "./transform.js";
import { Node } from "./node.js";
import { Scene } from "./scene.js";
import { Renderer } from "./renderer.js";
import { Cube } from "./cube.js";
import { Sphere } from "./sphere.js";

let canvas, device, context, renderer, camera, scene, shader, clip;

// cor de fundo; a fog usa a mesma cor para os objetos "sumirem" no fundo
const BACKGROUND = [0.87, 0.89, 0.93, 1.0];
const TABLE_TOP = 1.12;                          // altura do tampo (y)
const SPOT_POS = gm.vec3(-1.2, 3.4, 1.4);
const LIGHT2_POS = gm.vec3(3.0, 3.0, -2.0);
const TABLE_CENTER = gm.vec3(0.0, TABLE_TOP, 0.0);

// estado dos efeitos (teclas)
const fx = { fog: false, clip: false, light2: false, bump: true, projector: false };

// cria uma Transform: translada, gira em y (graus) e escala.
// Como a Transform multiplica pela direita, a escala e aplicada primeiro no
// objeto, depois a rotacao e por ultimo a translacao (T * R * S).
function trf (t, s = [1, 1, 1], rotY = 0) {
  const tr = new Transform();
  tr.translate(t[0], t[1], t[2]);
  if (rotY !== 0) tr.rotate(rotY, 0, 1, 0);
  tr.scale(s[0], s[1], s[2]);
  return tr;
}

async function initialize (targetFormat) {
  // camera + arcball
  // A camera define a matriz de vista (lookAt: olho, centro, up) e a de
  // projecao perspectiva. O arcball gira a cena em torno de um pivo que fica
  // a distancia olho-centro, por isso o centro tem que ser definido antes.
  camera = new Camera3D(2.6, 3.2, 5.2);
  camera.setCenter(0.0, 1.2, 0.0); // tem que vir antes do arcball
  camera.createArcball().attach(canvas);

  // spot apontando para o centro da mesa (cone de 26 graus).
  // O ambiente da luz e somado em todos os pontos, mesmo fora do cone; sem
  // ele, o que esta fora do spot ficaria totalmente preto.
  const spot = new SpotLight(SPOT_POS, TABLE_CENTER, 26, "world");
  spot.setAmbient(0.25, 0.25, 0.27);
  spot.setDiffuse(0.9, 0.9, 0.85);

  // Um shader so para a cena toda, assim todos os efeitos podem ser
  // combinados. space "world": a iluminacao e feita em coordenadas de mundo,
  // o mesmo espaco em que definimos as luzes, o projetor e o plano de corte.
  shader = await Shader.load(device, "../shaders/cena.wgsl", { light: spot, space: "world" });
  // Cada atributo do vertice vem num buffer separado, na mesma ordem dos
  // slots que Cube e Sphere usam: 0 posicao, 1 normal, 2 tangente, 3 uv.
  shader.setVertexBuffers([
    { arrayStride: 3 * 4, stepMode: "vertex", attributes: [{ format: "float32x3", offset: 0, varName: "coord" }] },
    { arrayStride: 3 * 4, stepMode: "vertex", attributes: [{ format: "float32x3", offset: 0, varName: "normal" }] },
    { arrayStride: 3 * 4, stepMode: "vertex", attributes: [{ format: "float32x3", offset: 0, varName: "tangent" }] },
    { arrayStride: 2 * 4, stepMode: "vertex", attributes: [{ format: "float32x2", offset: 0, varName: "texcoord" }] },
  ]);
  // cullMode none para ver o interior dos objetos com o plano de corte:
  // quando um objeto e cortado, enxergamos as faces de tras dele por dentro.
  const pipeline = new Pipeline(shader, targetFormat, { primitive: { cullMode: "none" } });

  // texturas
  const [madeira, terra, golfNormal, slide] = await Promise.all([
    Texture.load(device, "decal_texture", "../images/madeira.png"),
    Texture.load(device, "decal_texture", "../images/terra_512.png"),
    // O mapa de normais guarda vetores, nao cores. Se fosse lido como sRGB a
    // GPU aplicaria a correcao de gama e os vetores sairiam tortos, entao ele
    // e carregado em formato linear.
    Texture.load(device, "normal_texture", "../images/golfball.png", { format: "rgba8unorm" }),
    Texture.load(device, "slide_texture", "../images/smile_face.jpg"),
  ]);
  // texturas 1x1 para objetos sem decal / sem bump.
  // Branco nao altera a cor do material (multiplica por 1), e (0.5, 0.5, 1)
  // vira a normal (0, 0, 1) depois do "* 2 - 1", ou seja, nao perturba nada.
  const branco = new Texture(device, "decal_texture", { texel: [1, 1, 1] });
  const normalPlano = new Texture(device, "normal_texture", { texel: [0.5, 0.5, 1.0], format: "rgba8unorm" });
  const decalSampler = new Sampler(device, "decal_sampler");
  // clamp-to-edge para o slide nao se repetir fora da area projetada
  const slideSampler = new Sampler(device, "slide_sampler", { addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });

  // o TextureSet precisa ter as 5 texturas/samplers do shader
  const texSet = (decal, normal) => {
    const ts = new TextureSet([decal, decalSampler, normal, slide, slideSampler]);
    shader.addTextureSet(ts);
    return ts;
  };
  const tsMadeira = texSet(madeira, normalPlano);
  const tsTerra = texSet(terra, normalPlano);
  const tsLiso = texSet(branco, normalPlano);
  const tsGolf = texSet(branco, golfNormal);

  // materiais (modelo de Phong): base_color e a cor difusa, specular_color a
  // cor do brilho e shininess o expoente (quanto maior, menor e mais
  // concentrado o reflexo). "bump" e um campo nosso, so a bola usa.
  const mat = (r, g, b, extra = {}) => {
    const m = new PhongMaterial(r, g, b);
    for (const k in extra) m.set(k, extra[k]);
    shader.addMaterial(m);
    return m;
  };
  const madeiraMat = mat(1.0, 1.0, 1.0);
  madeiraMat.setSpecular(0.3, 0.3, 0.3);
  const amarelo = mat(0.85, 0.62, 0.12);
  const globo = mat(1.0, 1.0, 1.0);
  const vermelho = mat(0.75, 0.1, 0.1, { bump: 1.0 });
  vermelho.setShininess(64.0);

  // uma instancia de cada forma, reusada em todos os nos (instanciacao):
  // todas as pecas usam os mesmos buffers, so muda a transformacao do no
  const cube = new Cube(device);     // x,z em [-0.5,0.5], y em [0,1]
  const sphere = new Sphere(device); // raio 1, centro na origem

  // grafo de cena
  // A matriz final de cada objeto e o produto das transformacoes do caminho
  // da raiz ate ele. Material e texturas tambem sao herdados pelos filhos.
  // Como o cubo comeca em y = 0, para apoiar uma peca na outra basta
  // transladar ate a altura certa.
  const legs = [[-1.3, -0.8], [1.3, -0.8], [-1.3, 0.8], [1.3, 0.8]].map(([x, z]) =>
    new Node({ trf: trf([x, 0.0, z], [0.15, 1.0, 0.15]), shps: [cube] }));

  const mesa = new Node({
    apps: [madeiraMat, tsMadeira], // herdado pelo tampo e pelas pernas
    nodes: [
      new Node({ trf: trf([0.0, 1.0, 0.0], [3.0, TABLE_TOP - 1.0, 2.0]), shps: [cube] }), // tampo
      ...legs,
    ],
  });

  // os objetos ficam num no transladado ate a altura do tampo, entao as
  // posicoes abaixo sao relativas a mesa
  const objetos = new Node({
    trf: trf([0.0, TABLE_TOP, 0.0]),
    nodes: [
      // caixa amarela
      new Node({ trf: trf([-0.5, 0.0, -0.1], [1.0, 0.4, 0.6], 10), apps: [amarelo, tsLiso], shps: [cube] }),
      // globo terrestre sobre a caixa. E irmao da caixa e nao filho: se fosse
      // filho herdaria a escala (1, 0.4, 0.6) e ficaria achatado.
      new Node({ trf: trf([-0.5, 0.65, -0.1], [0.25, 0.25, 0.25]), apps: [globo, tsTerra], shps: [sphere] }),
      // bola vermelha com bump
      new Node({ trf: trf([0.8, 0.45, 0.25], [0.45, 0.45, 0.45]), apps: [vermelho, tsGolf], shps: [sphere] }),
    ],
  });

  scene = new Scene(new Node({ pipeline, nodes: [mesa, objetos] }));

  // extras
  // Plano ax + by + cz + d = 0 guardado como (a, b, c, d). O shader descarta
  // os pontos com dot(plano, (p, 1)) < 0. Com (0,0,0,1) o produto da sempre 1.
  clip = new ClipPlane(shader, 0, 0, 0, 1); // (0,0,0,1) nao corta nada

  shader.setValue("fog_color", gm.vec4(...BACKGROUND));
  shader.setValue("fog_density", 0.12);

  // O shader so aceita uma "light" (que ja e o spot), entao a 2a luz vai
  // direto nos campos do shader. Ela esta fixa no mundo, nao precisa mudar
  // de espaco.
  shader.setValue("light2_position", gm.vec4(LIGHT2_POS[0], LIGHT2_POS[1], LIGHT2_POS[2], 1.0));
  shader.setValue("light2_diffuse", gm.vec3(0.45, 0.5, 0.6));
  shader.setValue("light2_specular", gm.vec3(0.5, 0.5, 0.6));

  // projetor na mesma posicao do spot. Como ele nao se move, a matriz de
  // textura e calculada uma vez so.
  const projector = new Projector(SPOT_POS, TABLE_CENTER, 34);
  shader.setValue("tex_matrix", projector.getTexMatrix());

  applyEffects();
}

// atualiza os uniforms do shader de acordo com fx.
// Os uniforms do framework nao aceitam bool, entao os liga/desliga vao
// como 0 ou 1 num vec4 (x fog, y luz 2, z bump, w projetor).
function applyEffects () {
  shader.setValue("flags", gm.vec4(fx.fog ? 1 : 0, fx.light2 ? 1 : 0, fx.bump ? 1 : 0, fx.projector ? 1 : 0));
  if (fx.clip) clip.setPlane(0, 0, -1, 0.2); // mantem z <= 0.2
  else clip.setPlane(0, 0, 0, 1);
}

function onKey (event) {
  switch (event.key.toLowerCase()) {
    case "f": fx.fog = !fx.fog; break;
    case "c": fx.clip = !fx.clip; break;
    case "l": fx.light2 = !fx.light2; break;
    case "b": fx.bump = !fx.bump; break;
    case "p": fx.projector = !fx.projector; break;
    default: return;
  }
  applyEffects();
}

// Laco de desenho: o Renderer percorre o grafo de cena (empilhando as
// transformacoes de cada no) e manda os comandos para a GPU.
// O CSS so muda o tamanho em que o canvas aparece na tela; a resolucao em
// que a GPU desenha e width/height. Se os dois forem diferentes a imagem fica
// esticada ou borrada, entao a cada quadro a resolucao acompanha o tamanho da
// janela (vezes devicePixelRatio, para telas de alta densidade). O Renderer
// recria o depth buffer sozinho e a camera usa a nova proporcao na projecao.
function resizeCanvas () {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function draw () {
  resizeCanvas();
  renderer.render(context.getCurrentTexture(), scene, camera);
  requestAnimationFrame(draw);
}

// adapter = placa de video, device = conexao com ela (por onde criamos
// buffers, texturas e pipelines), context = ligacao do canvas com o WebGPU
async function main () {
  canvas = document.getElementById("canvas");
  const adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();
  context = canvas.getContext("webgpu");
  const targetFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: targetFormat });

  // depthTest liga o z-buffer, para o objeto mais proximo ficar na frente
  renderer = new Renderer(device, { depthTest: true, clearValue: BACKGROUND });
  await initialize(targetFormat);

  window.addEventListener("keydown", onKey);
  requestAnimationFrame(draw);
}

main();
