// Projeto 2 - shader da cena (Phong por fragmento, espaco de mundo)
// flags: x = fog, y = luz 2, z = bump, w = projetor
//
// Iluminacao por fragmento: o vertex shader so transforma e repassa posicao,
// normal etc.; a GPU interpola esses valores entre os vertices e o modelo de
// Phong e calculado no fragment shader, para cada pixel. Assim o brilho
// especular fica redondo mesmo numa malha com poucos triangulos.

struct Matrix {
  vertex: mat4x4<f32>,
  normal: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

// Os nomes dos campos importam: o framework so escreve os campos que
// existem aqui (luz, setValue, ClipPlane...). bool nao e aceito, por isso
// os liga/desliga ficam em flags como 0.0 / 1.0.
struct Global {
  projection: mat4x4<f32>,
  camera_position: vec4<f32>,
  // luz 1 (SpotLight)
  light_ambient: vec3<f32>,
  light_diffuse: vec3<f32>,
  light_specular: vec3<f32>,
  light_position: vec4<f32>,
  spot_direction: vec3<f32>,
  spot_cos_cutoff: f32,
  // luz 2 (pontual, espaco de mundo)
  light2_diffuse: vec3<f32>,
  light2_specular: vec3<f32>,
  light2_position: vec4<f32>,
  // plano de corte
  clip_plane: vec4<f32>,
  clip_plane_color: vec4<f32>, // usado pela classe ClipPlane
  // fog
  fog_color: vec4<f32>,
  fog_density: f32,
  // projetor
  tex_matrix: mat4x4<f32>,
  flags: vec4<f32>,
}
@group(0) @binding(0) var<uniform> global: Global;

struct Material {
  base_color: vec3<f32>,
  opacity: f32,
  specular_color: vec3<f32>,
  shininess: f32,
  bump: f32,
}
@group(2) @binding(0) var<uniform> material: Material;

@group(3) @binding(0) var decal_texture: texture_2d<f32>;
@group(3) @binding(1) var decal_sampler: sampler;
@group(3) @binding(2) var normal_texture: texture_2d<f32>;
@group(3) @binding(3) var slide_texture: texture_2d<f32>;
@group(3) @binding(4) var slide_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) tangent: vec3<f32>,
  @location(3) texcoord: vec2<f32>,
  @location(4) proj_coord: vec4<f32>,
}

@vertex
fn vs_main (@builtin(instance_index) instance_index: u32, @location(0) coord: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) tangent: vec3<f32>, @location(3) texcoord: vec2<f32>) -> VertexOutput {
  let m = matrix[instance_index];
  // com space "world", m.vertex e a matriz de modelagem (objeto -> mundo)
  let wpos = m.vertex * vec4<f32>(coord, 1.0);
  var out: VertexOutput;
  out.world_pos = wpos.xyz;
  // A normal usa a inversa transposta da modelagem (m.normal): com escala
  // nao uniforme, transformar a normal pela propria matriz faria ela deixar
  // de ser perpendicular a superficie.
  out.normal = (m.normal * vec4<f32>(normal, 0.0)).xyz;
  // ja a tangente fica "deitada" na superficie, entao segue a modelagem
  out.tangent = (m.vertex * vec4<f32>(tangent, 0.0)).xyz;
  out.texcoord = texcoord;
  // posicao do ponto na "imagem" do projetor (a divisao por w e feita no
  // fragment shader, depois da interpolacao)
  out.proj_coord = global.tex_matrix * wpos;
  // projection ja e Proj * View, porque a iluminacao e em espaco de mundo
  out.clip_position = global.projection * wpos;
  return out;
}

// 1 dentro do cone do spot, 0 fora.
// L aponta do ponto para a luz, entao -L aponta da luz para o ponto; o
// produto escalar com o eixo do cone e o cosseno do angulo entre eles.
fn spot_factor (L: vec3<f32>) -> f32 {
  let s = dot(-L, normalize(global.spot_direction));
  return select(0.0, 1.0, s >= global.spot_cos_cutoff);
}

// 1 dentro do slide, 0 fora (nao tem clamp-to-border no WebGPU).
// pc.w > 0 garante que o ponto esta na frente do projetor; sem isso a
// imagem tambem apareceria "atras" dele, invertida.
fn projector_mask (pc: vec4<f32>, uv: vec2<f32>) -> f32 {
  let inside = pc.w > 0.0 && all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
  return select(0.0, 1.0, inside);
}

// normal do mapa de normais (espaco tangente -> mundo).
// O mapa guarda a normal num sistema local da superficie: x na direcao da
// tangente T, y na bitangente B e z na normal N. Montando a base (T, B, N)
// em coordenadas de mundo, basta combinar os tres eixos com os valores
// lidos da textura.
fn bump_normal (N: vec3<f32>, tangent: vec3<f32>, nmap: vec3<f32>) -> vec3<f32> {
  // depois da interpolacao T pode nao ser mais perpendicular a N, entao
  // tiramos de T a componente na direcao de N (Gram-Schmidt)
  let T = normalize(tangent - N * dot(N, tangent));
  let B = cross(N, T);
  // a textura guarda valores em [0,1]; o vetor original estava em [-1,1]
  let nt = nmap * 2.0 - 1.0;
  return normalize(T * nt.x + B * nt.y + N * nt.z);
}

// difusa + especular (modelo de Phong).
// Difusa (Lambert): proporcional a N.L, o cosseno entre a normal e a luz.
// Especular: proporcional a (R.V)^shininess, onde R e a luz refletida em
// torno da normal; quanto mais perto o olho esta de R, mais brilho.
fn phong (N: vec3<f32>, L: vec3<f32>, V: vec3<f32>, kd: vec3<f32>, diffuse: vec3<f32>, specular: vec3<f32>) -> vec3<f32> {
  let ndotl = dot(N, L);
  // luz chegando por tras da superficie: sem difusa e sem especular
  if (ndotl <= 0.0) {
    return vec3<f32>(0.0);
  }
  let R = normalize(reflect(-L, N));
  let spec = pow(max(0.0, dot(R, V)), material.shininess);
  return kd * diffuse * ndotl + material.specular_color * specular * spec;
}

// Fog exponencial quadratica: f = exp(-(densidade * d)^2) e quanto da cor
// original sobra a distancia d da camera (1 perto, tende a 0 longe).
// Com a fog desligada (flags.x = 0) o fator vira 1 e a cor nao muda.
fn apply_fog (color: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
  let d = distance(global.camera_position.xyz, world_pos);
  let f = exp(-pow(global.fog_density * d, 2.0));
  return mix(global.fog_color.rgb, color, mix(1.0, f, global.flags.x));
}

@fragment
fn fs_main (in: VertexOutput) -> @location(0) vec4<f32> {
  // textureSample nao pode ficar dentro de if, entao le tudo antes.
  // (ele precisa comparar com os pixels vizinhos para escolher o filtro,
  // entao todos os pixels do grupo tem que executar a leitura juntos)
  let decal = textureSample(decal_texture, decal_sampler, in.texcoord).rgb;
  let nmap = textureSample(normal_texture, decal_sampler, in.texcoord).xyz;
  // divisao perspectiva do projetor: (s, t, _, q) -> (s/q, t/q)
  let uv = in.proj_coord.xy / in.proj_coord.w;
  let slide = textureSampleLevel(slide_texture, slide_sampler, uv, 0.0).rgb;

  // plano de corte: o WGSL nao tem gl_ClipDistance, entao o pixel do lado
  // negativo do plano e simplesmente descartado
  if (dot(global.clip_plane, vec4<f32>(in.world_pos, 1.0)) < 0.0) {
    discard;
  }

  // decal: a textura modula a cor difusa do material
  let kd = material.base_color * decal;

  // a interpolacao desnormaliza a normal, por isso o normalize
  var N = normalize(in.normal);
  if (global.flags.z * material.bump > 0.5) {
    N = bump_normal(N, in.tangent, nmap);
  }
  let V = normalize(global.camera_position.xyz - in.world_pos);

  // ambiente: luz "espalhada" que chega em todo lugar, somada uma vez so
  var color = kd * global.light_ambient;

  // luz 1: spot ou projetor.
  // c1 e a "cor" da luz que chega no ponto: o fator do cone no spot, ou a
  // cor do slide no projetor (que assim substitui o spot).
  let L1 = normalize(global.light_position.xyz - in.world_pos);
  var c1 = vec3<f32>(spot_factor(L1));
  if (global.flags.w > 0.5) {
    c1 = slide * projector_mask(in.proj_coord, uv);
  }
  color += phong(N, L1, V, kd, global.light_diffuse * c1, global.light_specular * c1);

  // luz 2: pontual, sem cone. As contribuicoes das luzes se somam.
  if (global.flags.y > 0.5) {
    let L2 = normalize(global.light2_position.xyz - in.world_pos);
    color += phong(N, L2, V, kd, global.light2_diffuse, global.light2_specular);
  }

  return vec4<f32>(apply_fog(color, in.world_pos), 1.0);
}
