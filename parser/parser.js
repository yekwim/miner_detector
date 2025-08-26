// parser_v2.js — normalização + textos para embeddings (sem chunking)
// Uso: node parser_v2.js <crawl_output_dir> [--out=dir] [--no-ast]
//
// Saídas:
//  - <out>/embedding_texts.jsonl         (por arquivo; texto normalizado para embeddings)
//  - <out>/norm/<sha|base>.js.tokens.txt (tokens semânticos do JS)
//  - <out>/norm/<sha|base>.wat           (WAT normalizado)
//  - <out>/norm/<sha|base>.wat.opseq.txt (sequência de opcodes do WAT)
//  - <out>/ast/<sha|base>.js.ast.json    (opcional)
//  - <out>/summary.json

const fs = require('fs').promises;
const fssync = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const esprima = require('esprima');
const estraverse = require('estraverse');

// -------------- util --------------
const exists = async p => !!(await fs.stat(p).catch(()=>false));
async function mkdirp(p){ await fs.mkdir(p, {recursive: true}); }
function nowIso(){ return new Date().toISOString(); }
function readLines(filePath){
  const data = fssync.readFileSync(filePath, 'utf8');
  return data.split(/\r?\n/).filter(Boolean).map(l => JSON.parse(l));
}
function sanitizeBase(u){
  try {
    const base = path.basename(u || 'resource');
    return base.replace(/\.[^.]*$/, '').replace(/[^a-zA-Z0-9._-]/g, '_');
  } catch { return 'resource'; }
}

// -------------- JS → tokens semânticos --------------
function memberToStr(m){
  if(!m) return '<?>'
  if(m.type === 'Identifier') return m.name;
  if(m.type === 'Literal') return String(m.value);
  if(m.type === 'MemberExpression'){
    return `${memberToStr(m.object)}.${memberToStr(m.property)}`
  }
  return m.type;
}
function jsToSemanticTokens(jsCode){
  const code = jsCode
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/[^\n\r]*/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  const toks = [];
  let ast;
  try{
    ast = esprima.parseScript(code, { tolerant: true });
  }catch{
    // fallback: devolve código compactado
    return code;
  }
  estraverse.traverse(ast, {
    enter(node){
      toks.push(node.type);
      if(node.type === 'CallExpression'){
        if(node.callee?.type === 'Identifier'){
          toks.push(`Call(${node.callee.name})`);
        } else if (node.callee?.type === 'MemberExpression'){
          toks.push(`Call(${memberToStr(node.callee.object)}.${memberToStr(node.callee.property)})`);
        }
      }
      if(node.type === 'NewExpression' && node.callee){
        const name = node.callee.type === 'Identifier' ? node.callee.name : memberToStr(node.callee);
        toks.push(`New(${name})`);
      }
    }
  });
  if(/WebAssembly\.(instantiate|compile)/.test(code)) toks.push('Flag(WebAssembly)');
  if(/WebSocket\s*\(/.test(code)) toks.push('Flag(WebSocket)');
  if(/new\s+Worker\s*\(/.test(code)) toks.push('Flag(Worker)');
  if(/postMessage\s*\(/.test(code)) toks.push('Flag(postMessage)');
  return toks.join(' ').trim();
}

// -------------- WASM → WAT → opcodes --------------
function normalizeWatText(wat){
  let t = wat.replace(/\(\;[\s\S]*?\;\)/g, '').replace(/;;.*$/gm, '');
  t = t.replace(/\$[A-Za-z_][\w\-]*/g, '$id'); // nomes
  t = t.replace(/([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)/g, 'NUM'); // números
  t = t.replace(/\s+/g, ' ').trim();
  return t;
}
function watToOpcodeTokens(watText){
  const opRegex = /\b(i32\.\w+|i64\.\w+|f32\.\w+|f64\.\w+|local\.(?:get|set|tee)|global\.(?:get|set)|memory\.(?:grow|size)|call|call_indirect|block|loop|if|else|end|br_if|br|return|select|drop|unreachable)\b/g;
  const out = [];
  let m; while((m = opRegex.exec(watText)) !== null) out.push(m[0]);
  return out.join(' ');
}
async function wasmToWat(wasmPath){
  async function tryTool(cmd, args){
    return await new Promise(res=>{
      const p = spawn(cmd, args);
      let out = ''; p.stdout.on('data', d=> out += String(d));
      p.on('close', c => res(c===0 ? out : null));
      p.on('error', ()=> res(null));
    });
  }
  let wat = await tryTool('wasm-tools', ['print', wasmPath]);
  if(wat) return wat;
  return await tryTool('wasm2wat', [wasmPath]);
}

// -------------- manifest fallback --------------
async function fallbackEnumerate(jsDir, wasmDir, workersDir){
  const list = [];
  if(await exists(jsDir)){
    for (const f of await fs.readdir(jsDir)) if (f.endsWith('.js'))
      list.push({ type: 'js', path: path.join(jsDir, f), url: '', sha256: null });
  }
  if(await exists(workersDir)){
    for (const f of await fs.readdir(workersDir)) if (f.endsWith('.js'))
      list.push({ type: 'js', path: path.join(workersDir, f), url: '', sha256: null });
  }
  if(await exists(wasmDir)){
    for (const f of await fs.readdir(wasmDir)) if (f.endsWith('.wasm'))
      list.push({ type: 'wasm', path: path.join(wasmDir, f), url: '', sha256: null });
  }
  return list;
}

// -------------- main --------------
async function main(){
  const crawlDir = process.argv[2];
  if(!crawlDir){
    console.error('Uso: node parser_v2.js <crawl_output_dir> [--out=dir] [--no-ast]');
    process.exit(1);
  }
  const args = process.argv.slice(3);
  const getArg = (k, d) => {
    const a = args.find(x => x.startsWith(k+'='));
    return a ? a.split('=')[1] : d;
  };
  const outDir = getArg('--out', path.join(process.cwd(), 'data', 'parser_output'));
  const genAST = !args.includes('--no-ast');

  await mkdirp(outDir);
  await mkdirp(path.join(outDir, 'norm'));
  if(genAST) await mkdirp(path.join(outDir, 'ast'));

  const manifestPath = path.join(crawlDir, 'manifest.jsonl');
  const jsDir = path.join(crawlDir, 'js');
  const wasmDir = path.join(crawlDir, 'wasm');
  const workersDir = path.join(crawlDir, 'workers');

  const items = fssync.existsSync(manifestPath)
    ? readLines(manifestPath)
    : await fallbackEnumerate(jsDir, wasmDir, workersDir);

  const outJSONL = path.join(outDir, 'embedding_texts.jsonl');
  const stream = fssync.createWriteStream(outJSONL, { flags: 'w' });

  const stats = { js: 0, wasm: 0, ast: 0, errors: 0, written: 0 };

  for (const rec of items){
    try{
      if(rec.type === 'js'){
        const text = await fs.readFile(rec.path, 'utf8');
        const tokens = jsToSemanticTokens(text);
        const base = rec.sha256 || sanitizeBase(rec.url || rec.path);
        await fs.writeFile(path.join(outDir, 'norm', `${base}.js.tokens.txt`), tokens, 'utf8');
        // JSONL para embeddings
        stream.write(JSON.stringify({
          id: base,
          sha256: rec.sha256 || null,
          type: 'js',
          url: rec.url || '',
          path: rec.path,
          text: tokens
        })+'\n');
        stats.js++; stats.written++;
        if(genAST){
          try{
            const ast = esprima.parseScript(text, { tolerant: true });
            await fs.writeFile(path.join(outDir, 'ast', `${base}.js.ast.json`), JSON.stringify(ast), 'utf8');
            stats.ast++;
          }catch{}
        }
      }
      if(rec.type === 'wasm'){
        const wat = await wasmToWat(rec.path);
        if(!wat) continue;
        const watNorm = normalizeWatText(wat);
        const opseq = watToOpcodeTokens(watNorm);
        const base = rec.sha256 || sanitizeBase(rec.url || rec.path);
        await fs.writeFile(path.join(outDir, 'norm', `${base}.wat`), watNorm, 'utf8');
        await fs.writeFile(path.join(outDir, 'norm', `${base}.wat.opseq.txt`), opseq, 'utf8');
        stream.write(JSON.stringify({
          id: base,
          sha256: rec.sha256 || null,
          type: 'wasm',
          url: rec.url || '',
          path: rec.path,
          text: opseq
        })+'\n');
        stats.wasm++; stats.written++;
      }
    }catch(e){
      stats.errors++; console.warn(`Falha em ${rec.path}: ${e.message}`);
    }
  }

  stream.end();
  await fs.writeFile(path.join(outDir, 'summary.json'),
    JSON.stringify({ generated_at: nowIso(), input_crawl_dir: crawlDir, output_dir: outDir, stats }, null, 2),
    'utf8'
  );

  console.log('Parser v2 concluído.');
  console.log(`JS: ${stats.js} WASM: ${stats.wasm} escritos: ${stats.written} AST: ${stats.ast} ERR: ${stats.errors}`);
  console.log(`JSONL para embeddings: ${outJSONL}`);
}

if(require.main === module){
  main().catch(err=>{ console.error(err); process.exit(1); });
}

module.exports = { jsToSemanticTokens, normalizeWatText, watToOpcodeTokens };
