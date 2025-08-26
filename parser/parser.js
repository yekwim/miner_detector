// Parser v2 — normalização + tokens + chunking (300–800) + JSONL
// Uso: node parser_v2.js <crawl_output_dir> [--out=dir] [--min=300] [--max=800] [--overlap=80] [--no-ast]

const fs = require('fs').promises;
const fssync = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const esprima = require('esprima');
const estraverse = require('estraverse');
const { encode, decode } = require('gpt-tokenizer');

// ------------------------ util ------------------------
const exists = async p => !!(await fs.stat(p).catch(()=>false));
const mkdirp = p => fs.mkdir(p, { recursive: true });

function readLines(filePath) {
  const data = fssync.readFileSync(filePath, 'utf8');
  return data.split(/\r?\n/).filter(Boolean).map(l => JSON.parse(l));
}

function nowIso() { return new Date().toISOString(); }

function sanitizeBase(u) {
  try {
    const base = path.basename(u || 'resource');
    return base.replace(/\.[^.]*$/, '').replace(/[^a-zA-Z0-9._-]/g, '_');
  } catch { return 'resource'; }
}

// ------------------------ chunking ------------------------
function chunkByTokens(text, { minT=300, maxT=800, overlap=80 } = {}) {
  const ids = encode(text); // GPT BPE
  const chunks = [];
  if (ids.length === 0) return [];
  let start = 0;
  while (start < ids.length) {
    let end = Math.min(start + maxT, ids.length);
    if (end - start < minT && end < ids.length) {
      end = Math.min(start + minT, ids.length);
    }
    const slice = ids.slice(start, end);
    chunks.push({
      n_tokens: slice.length,
      text: decode(Uint32Array.from(slice)),
    });
    if (end === ids.length) break;
    start = end - Math.min(overlap, end); // desliza com overlap
  }
  return chunks;
}

// ------------------------ JS normalização → tokens ------------------------
function jsToSemanticTokens(jsCode) {
  // Remove comentários e espaços extras rápido, sem quebrar semântica
  const code = jsCode
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/[^\n\r]*/g, '')
    .replace(/\s+/g, ' ')
    .trim();

  // AST
  const ast = esprima.parseScript(code, { tolerant: true, range: true, loc: false });

  // Caminho pré-ordem, registrando tipos de nó e algumas chamadas sensíveis
  const toks = [];
  estraverse.traverse(ast, {
    enter(node) {
      toks.push(node.type);
      // Chamadas sensíveis
      if (node.type === 'CallExpression') {
        // nome simples
        if (node.callee && node.callee.type === 'Identifier') {
          toks.push(`Call(${node.callee.name})`);
        }
        // obj.prop
        if (node.callee && node.callee.type === 'MemberExpression') {
          const obj = memberToStr(node.callee.object);
          const prop = memberToStr(node.callee.property);
          toks.push(`Call(${obj}.${prop})`);
        }
      }
      // New Worker, WebSocket etc
      if (node.type === 'NewExpression' && node.callee) {
        const name = node.callee.type === 'Identifier'
          ? node.callee.name
          : memberToStr(node.callee);
        toks.push(`New(${name})`);
      }
    }
  });

  // Heurísticas de orquestração típicas
  const shallowFlags = [];
  if (/WebAssembly\.(instantiate|compile)/.test(code)) shallowFlags.push('WebAssembly');
  if (/WebSocket\s*\(/.test(code)) shallowFlags.push('WebSocket');
  if (/new\s+Worker\s*\(/.test(code)) shallowFlags.push('Worker');
  if (/postMessage\s*\(/.test(code)) shallowFlags.push('postMessage');
  toks.push(...shallowFlags.map(f => `Flag(${f})`));

  // Sequência final
  return toks.join(' ');
}

function memberToStr(m) {
  if (!m) return '<?>';

  if (m.type === 'Identifier') return m.name;
  if (m.type === 'Literal') return String(m.value);

  if (m.type === 'MemberExpression') {
    const left = memberToStr(m.object);
    const right = memberToStr(m.property);
    return `${left}.${right}`;
  }
  return m.type;
}

// ------------------------ WAT normalização → opcodes ------------------------
function normalizeWatText(wat) {
  // remove comentários (; ;) e ;; linha
  let t = wat.replace(/\(\;[\s\S]*?\;\)/g, '').replace(/;;.*$/gm, '');
  // normaliza nomes $foo → $id
  t = t.replace(/\$[A-Za-z_][\w\-]*/g, '$id');
  // números → NUM
  t = t.replace(/([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)/g, 'NUM');
  // espaços
  t = t.replace(/\s+/g, ' ').trim();
  return t;
}

function watToOpcodeTokens(watText) {
  const t = watText;
  const opRegex = /\b(i32\.\w+|i64\.\w+|f32\.\w+|f64\.\w+|local\.(?:get|set|tee)|global\.(?:get|set)|memory\.(?:grow|size)|call|call_indirect|block|loop|if|else|end|br_if|br|return|select|drop|unreachable)\b/g;
  const out = [];
  let m;
  while ((m = opRegex.exec(t)) !== null) out.push(m[0]);
  return out.join(' ');
}

// ------------------------ WASM→WAT ------------------------
async function wasmToWat(wasmPath) {
  // tenta wasm-tools
  const tryTool = (cmd, args) => new Promise((resolve) => {
    const p = spawn(cmd, args);
    let out = '', err = '';
    p.stdout.on('data', d => out += String(d));
    p.stderr.on('data', d => err += String(d));
    p.on('close', code => resolve(code === 0 ? out : null));
    p.on('error', () => resolve(null));
  });

  let wat = await tryTool('wasm-tools', ['print', wasmPath]);
  if (wat) return wat;
  wat = await tryTool('wasm2wat', [wasmPath]);
  return wat; // pode ser null
}

// ------------------------ processamento principal ------------------------
async function main() {
  const crawlDir = process.argv[2];
  if (!crawlDir) {
    console.error('Uso: node parser_v2.js <crawl_output_dir> [--out=dir] [--min=300] [--max=800] [--overlap=80] [--no-ast]');
    process.exit(1);
  }

  const args = process.argv.slice(3);
  const getArg = (k, d) => {
    const a = args.find(x => x.startsWith(k + '='));
    return a ? a.split('=')[1] : d;
  };

  const outDir = getArg('--out', path.join(process.cwd(), 'parser_output'));
  const minT = parseInt(getArg('--min', '300'), 10);
  const maxT = parseInt(getArg('--max', '800'), 10);
  const overlap = parseInt(getArg('--overlap', '80'), 10);
  const genAST = !args.includes('--no-ast');

  await mkdirp(outDir);
  await mkdirp(path.join(outDir, 'ast'));
  await mkdirp(path.join(outDir, 'norm'));
  await mkdirp(path.join(outDir, 'chunks'));

  const manifestPath = path.join(crawlDir, 'manifest.jsonl');
  const jsDir = path.join(crawlDir, 'js');
  const wasmDir = path.join(crawlDir, 'wasm');
  const workersDir = path.join(crawlDir, 'workers');

  const items = fssync.existsSync(manifestPath)
    ? readLines(manifestPath)
    : await fallbackEnumerate(jsDir, wasmDir, workersDir);

  const chunksJSONL = path.join(outDir, 'chunks', 'embeddings_chunks.jsonl');
  const chunksStream = fssync.createWriteStream(chunksJSONL, { flags: 'w' });

  let stats = { js: 0, wasm: 0, chunks: 0, ast: 0, errors: 0 };

  for (const rec of items) {
    try {
      if (rec.type === 'js') {
        const text = await fs.readFile(rec.path, 'utf8');
        const tokens = jsToSemanticTokens(text);
        const base = rec.sha256 || sanitizeBase(rec.url || rec.path);
        await fs.writeFile(path.join(outDir, 'norm', `${base}.js.tokens.txt`), tokens, 'utf8');

        // chunks
        const chunks = chunkByTokens(tokens, { minT, maxT, overlap });
        chunks.forEach((c, i) => {
          chunksStream.write(JSON.stringify({
            id: `${base}::${i}`,
            sha256: rec.sha256 || null,
            type: 'js',
            url: rec.url || '',
            path: rec.path,
            chunk_index: i,
            n_tokens: c.n_tokens,
            text: c.text
          }) + '\n');
        });
        stats.js += 1;
        stats.chunks += chunks.length;

        if (genAST) {
          // AST opcional para auditoria
          const ast = esprima.parseScript(text, { tolerant: true, range: false, loc: false });
          await fs.writeFile(path.join(outDir, 'ast', `${base}.js.ast.json`), JSON.stringify(ast), 'utf8');
          stats.ast += 1;
        }
      }

      if (rec.type === 'wasm') {
        const wat = await wasmToWat(rec.path);
        if (!wat) continue;

        const watNorm = normalizeWatText(wat);
        const opseq = watToOpcodeTokens(watNorm);

        const base = rec.sha256 || sanitizeBase(rec.url || rec.path);
        await fs.writeFile(path.join(outDir, 'norm', `${base}.wat`), watNorm, 'utf8');
        await fs.writeFile(path.join(outDir, 'norm', `${base}.wat.opseq.txt`), opseq, 'utf8');

        // chunks
        const chunks = chunkByTokens(opseq, { minT, maxT, overlap });
        chunks.forEach((c, i) => {
          chunksStream.write(JSON.stringify({
            id: `${base}::${i}`,
            sha256: rec.sha256 || null,
            type: 'wasm',
            url: rec.url || '',
            path: rec.path,
            chunk_index: i,
            n_tokens: c.n_tokens,
            text: c.text
          }) + '\n');
        });
        stats.wasm += 1;
        stats.chunks += chunks.length;
      }

      // Workers salvos pelo crawler aparecem como type "js" no manifest, mas com path em /workers.
      // Não é preciso tratamento diferente aqui.

    } catch (e) {
      stats.errors += 1;
      console.warn(`Falha em ${rec.path}: ${e.message}`);
    }
  }

  chunksStream.end();

  // Relatório sintético
  const summary = {
    generated_at: nowIso(),
    input_crawl_dir: crawlDir,
    output_dir: outDir,
    params: { min_tokens: minT, max_tokens: maxT, overlap },
    stats
  };
  await fs.writeFile(path.join(outDir, 'summary.json'), JSON.stringify(summary, null, 2), 'utf8');

  console.log('Parser v2 concluído.');
  console.log(`JS: ${stats.js}  WASM: ${stats.wasm}  CHUNKS: ${stats.chunks}  AST: ${stats.ast}  ERR: ${stats.errors}`);
  console.log(`Chunks JSONL: ${chunksJSONL}`);
}

// fallback caso não exista manifest.jsonl
async function fallbackEnumerate(jsDir, wasmDir, workersDir) {
  const list = [];
  if (await exists(jsDir)) {
    for (const f of await fs.readdir(jsDir)) if (f.endsWith('.js'))
      list.push({ type: 'js', path: path.join(jsDir, f), url: '', sha256: null });
  }
  if (await exists(workersDir)) {
    for (const f of await fs.readdir(workersDir)) if (f.endsWith('.js'))
      list.push({ type: 'js', path: path.join(workersDir, f), url: '', sha256: null });
  }
  if (await exists(wasmDir)) {
    for (const f of await fs.readdir(wasmDir)) if (f.endsWith('.wasm'))
      list.push({ type: 'wasm', path: path.join(wasmDir, f), url: '', sha256: null });
  }
  return list;
}

if (require.main === module) {
  main().catch(err => { console.error(err); process.exit(1); });
}

module.exports = {
  jsToSemanticTokens,
  normalizeWatText,
  watToOpcodeTokens,
  chunkByTokens
};
