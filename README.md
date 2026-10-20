# Projeto 2 - Cena 3D: mesa e objetos (INF1761)

Cena 3D feita sobre o código-base da disciplina (JavaScript + WebGPU/WGSL).

## Como rodar

1. Na raiz do repositório (a pasta que contém `js/`, `shaders/` e `images/`), suba um servidor HTTP:
   ```bash
   python3 -m http.server 8000
   ```
2. Abra `http://localhost:8000/js/projeto2.html` num navegador com WebGPU (Chrome/Edge). No Linux, pode ser preciso ativar `chrome://flags/#enable-unsafe-webgpu`.

Abrir o arquivo direto (`file://`) não funciona, porque o navegador bloqueia o `fetch` dos shaders e das imagens.

## Controles

| Entrada | Ação |
|---|---|
| Arrastar o mouse | Gira a câmera (arcball) |
| `F` | Liga/desliga a fog |
| `C` | Liga/desliga o plano de corte |
| `L` | Liga/desliga a 2ª fonte de luz |
| `B` | Liga/desliga o bump mapping |
| `P` | Liga/desliga o projetor de slide (substitui o spot) |

## Arquivos do projeto

| Arquivo | Conteúdo |
|---|---|
| `js/projeto2.html`, `js/projeto2.js` | Ponto de entrada: monta a cena e trata o teclado |
| `js/spotlight.js` | Luz spot (nova) |
| `js/projector.js` | Projetor de slide (novo) |
| `js/texture.js` | Do código-base, com uma opção de formato linear para o mapa de normais |
| demais arquivos de `js/` | Código-base da disciplina (grafo de cena, câmera, arcball, shader, etc.) |
| `shaders/cena.wgsl` | Shader único da cena |
| `images/` | Texturas: madeira (mesa), terra (globo), golfball (mapa de normais), smile_face (slide) |
