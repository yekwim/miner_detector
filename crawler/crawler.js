// crawler.js - Versão 40 (recriada com melhorias)
// Uso:
//   node crawler.js <URL> [--external] [--external-limit=N] [--no-subs]
//                    [--allow=dom1,dom2] [--block=dom1,dom2]
//                    [--pages=N] [--depth=N] [--concurrent=N]
//                    [--max-bytes=26214400] [--headless]
const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');
const { URL } = require('url');
const { EventEmitter } = require('events');
const { createHash } = require('crypto');

class AdvancedCrawler extends EventEmitter {
  constructor(options = {}) {
    super();
    this.config = {
      maxDepth: 3,
      maxPages: 50,
      timeout: 30000,
      concurrentPages: 2,
      retryAttempts: 2,
      userAgent:
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      allowExternalDomains: false,
      allowSubdomains: true,
      maxExternalDepth: 1,
      maxExternalLinksPerDomain: 3,
      allowedDomains: [],
      blockedDomains: [],
      headless: 'new',
      maxResourceBytes: 25 * 1024 * 1024, // 25 MB
      ...options,
    };

    this.state = {
      activePages: 0,
      crawledCount: 0,
      visitedUrls: new Set(),
      failedUrls: new Set(),
      externalDomainCount: new Map(),
    };

    this.resources = {
      js: new Map(),
      wasm: new Map(),
      json: new Map(),
    };

    this.hashIndex = new Map(); // sha256 -> info
    this.manifest = [];
    this.pendingWasmRequests = new Map();

    this.browser = null;
    this.baseDir = null;
    this.sessionId = null;
    this.queue = null;
    this.baseUrl = null;
  }

  // ========= init / infra =========
  async init(targetUrl) {
    const parsedUrl = new URL(targetUrl);
    this.baseUrl = `${parsedUrl.protocol}//${parsedUrl.host}`;

    this.baseDir = await this.createOutputDir(targetUrl);
    this.sessionId = Date.now();

    this.browser = await puppeteer.launch({
      headless: this.config.headless,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-extensions',
        '--blink-settings=imagesEnabled=false',
        '--disable-background-networking',
        '--disable-default-apps',
        '--disable-sync',
        '--ignore-certificate-errors',
      ],
      executablePath: process.env.CHROMIUM_PATH || undefined,
    });

    this.queue = new PriorityQueue({
      comparator: (a, b) => {
        if (a.depth !== b.depth) return a.depth - b.depth;
        return a.priority - b.priority;
      },
    });

    this.queue.push({
      url: targetUrl,
      depth: 0,
      priority: 1,
      referrer: null,
      isExternal: false,
    });

    this.emit('initialized', { targetUrl, baseDir: this.baseDir });
  }

  async createOutputDir(targetUrl) {
    const domain = new URL(targetUrl).hostname.replace(/[^a-zA-Z0-9]/g, '_');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const dirName = `crawl_${domain}_${timestamp}`;
    const baseDir = path.join(process.cwd(), 'crawl_output', dirName);

    const dirs = ['html', 'js', 'wasm', 'workers', 'json'];
    await Promise.all(dirs.map((d) => fs.mkdir(path.join(baseDir, d), { recursive: true })));
    return baseDir;
  }

  // ========= loop principal =========
  async crawl() {
    const startTime = Date.now();
    this.emit('crawlStarted', { maxPages: this.config.maxPages });

    while (!this.queue.isEmpty() && this.state.crawledCount < this.config.maxPages) {
      if (this.state.activePages < this.config.concurrentPages) {
        this.processNextPage();
      } else {
        await this.sleep(100);
      }
      if (this.state.crawledCount > 0 && this.state.crawledCount % 10 === 0) {
        this.emit('progress', {
          crawled: this.state.crawledCount,
          queued: this.queue.size(),
          active: this.state.activePages,
        });
      }
    }
    while (this.state.activePages > 0) await this.sleep(100);

    const duration = Date.now() - startTime;
    await this.generateReport(duration);
    await this.cleanup();

    this.emit('crawlCompleted', {
      duration,
      pagesProcessed: this.state.crawledCount,
      resourcesFound: this.getTotalResourceCount(),
    });
  }

  async processNextPage() {
    if (this.queue.isEmpty()) return;
    const { url, depth, referrer, isExternal } = this.queue.pop();
    if (this.state.visitedUrls.has(url)) return;

    this.state.visitedUrls.add(url);
    this.state.activePages++;

    let page = null;
    try {
      page = await this.browser.newPage();
      await this.setupPage(page);
      await this.processPageWithRetry(page, url, depth, referrer, isExternal);
    } catch (error) {
      this.state.failedUrls.add(url);
      this.emit('pageError', { url, depth, isExternal, error: error.message });
    } finally {
      if (page) await page.close().catch(() => {});
      this.state.activePages--;
      this.state.crawledCount++;
    }
  }

  async setupPage(page) {
    page.setDefaultNavigationTimeout(this.config.timeout);
    page.setDefaultTimeout(this.config.timeout);
    await page.setUserAgent(this.config.userAgent);
    await page.setViewport({ width: 1366, height: 768 });

    await this.injectCaptureHooks(page);

    await page.setRequestInterception(true);
    page.on('request', this.handleRequest.bind(this));
    page.on('response', this.handleResponse.bind(this, page));

    const client = await page.target().createCDPSession();
    await this.setupCDPInterceptors(client);
  }

  // ========= CDP e injeções =========
  async setupCDPInterceptors(client) {
    try {
      await client.send('Network.enable');
      await client.send('Runtime.enable');

      client.on('Network.requestWillBeSent', (ev) => {
        const u = ev.request.url;
        if (this.looksLikeWasmUrl(u)) this.pendingWasmRequests.set(ev.requestId, { url: u, t: Date.now() });
      });

      client.on('Network.responseReceived', async (ev) => {
        const resp = ev.response;
        const u = resp.url;
        const wasTracked = this.pendingWasmRequests.get(ev.requestId);
        if (this.isWasmResponse(resp, u) || wasTracked) {
          try {
            const body = await client.send('Network.getResponseBody', { requestId: ev.requestId });
            const buffer = Buffer.from(body.body, body.base64Encoded ? 'base64' : 'binary');
            await this.saveWasmFromNetwork(u, buffer);
          } catch {}
          this.pendingWasmRequests.delete(ev.requestId);
        }
      });

      client.on('Network.loadingFinished', (ev) => this.pendingWasmRequests.delete(ev.requestId));
      client.on('Network.loadingFailed', (ev) => this.pendingWasmRequests.delete(ev.requestId));

      await client.send('Runtime.addBinding', { name: 'captureWasm' });
      await client.send('Runtime.addBinding', { name: 'captureCode' });

      client.on('Runtime.bindingCalled', async (event) => {
        try {
          if (event.name === 'captureWasm') {
            const wasmData = JSON.parse(event.payload);
            await this.saveDynamicWasm(wasmData);
          }
          if (event.name === 'captureCode') {
            const data = JSON.parse(event.payload); // { kind, code?, bytes?, url, context }
            const base = this.sanitizeUrlBase(data.url || 'dynamic');
            const meta = { timestamp: Date.now(), source: 'dynamic', context: data.context || '', url: data.url || '' };
            if (data.kind === 'wasm' && data.bytes) {
              const saved = await this.saveBufferUnique(Buffer.from(data.bytes), 'wasm', base, 'wasm', meta);
              this.emit('resourceSaved', { type: 'wasm', url: data.url || 'dynamic:wasm', path: saved.path });
            } else if (data.kind === 'js' && data.code) {
              const saved = await this.saveTextUnique(data.code, 'workers', base, 'js', meta);
              this.emit('resourceSaved', { type: 'js', url: data.url || 'dynamic:js', path: saved.path });
            }
          }
        } catch {}
      });
    } catch {
      // segue sem CDP se falhar
    }
  }

  async injectCaptureHooks(page) {
    await page.evaluateOnNewDocument(() => {
      const send = (name, payload) => { try { window[name](JSON.stringify(payload)); } catch {} };

      // Map blobURL -> {type, bytes?, text?}
      const __blobMap = new Map();

      // Blob wrapper para capturar conteúdo
      const OriginalBlob = window.Blob;
      window.Blob = function(parts, opts) {
        const b = new OriginalBlob(parts, opts);
        b.__rawParts = parts;
        b.__type = (opts && opts.type) || '';
        return b;
      };

      // URL.createObjectURL wrapper
      const OriginalCreateObjectURL = URL.createObjectURL;
      URL.createObjectURL = function(obj) {
        const url = OriginalCreateObjectURL.call(URL, obj);
        if (obj instanceof Blob) {
          const reader = new FileReader();
          reader.onload = () => {
            const u8 = new Uint8Array(reader.result);
            __blobMap.set(url, { type: (obj.__type || '').toLowerCase(), bytes: Array.from(u8), text: null });
            if (u8.length >= 4 && u8[0] === 0x00 && u8[1] === 0x61 && u8[2] === 0x73 && u8[3] === 0x6d) {
              send('captureCode', { kind: 'wasm', bytes: Array.from(u8), url, context: 'blob' });
            }
          };
          reader.readAsArrayBuffer(obj);
          obj.text?.().then(t => {
            const rec = __blobMap.get(url) || {};
            rec.text = t; __blobMap.set(url, rec);
            if (rec.text && (!rec.bytes || rec.bytes.length < 4)) {
              send('captureCode', { kind: 'js', code: rec.text, url, context: 'blob' });
            }
          }).catch(()=>{});
        }
        return url;
      };

      // Intercepta Worker criado via blob:
      const OriginalWorker = window.Worker;
      if (OriginalWorker) {
        window.Worker = function(scriptURL, options) {
          try {
            if (typeof scriptURL === 'string' && scriptURL.startsWith('blob:') && __blobMap.has(scriptURL)) {
              const rec = __blobMap.get(scriptURL);
              if (rec && rec.text) send('captureCode', { kind: 'js', code: rec.text, url: scriptURL, context: 'worker-blob' });
            }
          } catch {}
          return new OriginalWorker(scriptURL, options);
        };
      }

      function reportWasm(buffer, context, url) {
        try {
          const u8 = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
          if (u8.length >= 4 && u8[0] === 0x00 && u8[1] === 0x61 && u8[2] === 0x73 && u8[3] === 0x6d) {
            send('captureWasm', { buffer: Array.from(u8), timestamp: Date.now(), url: url || location.href, context, size: u8.length });
          }
        } catch {}
      }

      if (typeof WebAssembly !== 'undefined') {
        const _inst = WebAssembly.instantiate;
        const _instS = WebAssembly.instantiateStreaming;
        const _comp = WebAssembly.compile;
        const _compS = WebAssembly.compileStreaming;
        const _mod  = WebAssembly.Module;

        WebAssembly.instantiate = async function(b, i) { reportWasm(b, 'instantiate'); return _inst.call(this, b, i); };
        WebAssembly.compile    = async function(b)     { reportWasm(b, 'compile');    return _comp.call(this, b); };
        WebAssembly.Module     = function(b)           { reportWasm(b, 'Module');     return new _mod(b); };

        WebAssembly.instantiateStreaming = async function(src, i) {
          const res = await _instS.call(this, src, i);
          try { const c = src.clone?.(); if (c) reportWasm(await c.arrayBuffer(), 'instantiateStreaming'); } catch {}
          return res;
        };
        WebAssembly.compileStreaming = async function(src) {
          const res = await _compS.call(this, src);
          try { const c = src.clone?.(); if (c) reportWasm(await c.arrayBuffer(), 'compileStreaming'); } catch {}
          return res;
        };
      }

      // Copia bytes de Responses suspeitas
      const OrigArrayBuffer = Response.prototype.arrayBuffer;
      Response.prototype.arrayBuffer = async function() {
        const ct = (this.headers.get('content-type') || '').toLowerCase();
        const ab = await OrigArrayBuffer.call(this);
        if (ct.includes('application/wasm') || ct.includes('octet-stream')) {
          reportWasm(ab, 'Response.arrayBuffer', this.url || location.href);
        }
        return ab;
      };
    });
  }

  // ========= network handlers =========
  handleRequest(request) {
    const url = request.url();
    const type = request.resourceType();

    if (this.looksLikeWasmUrl(url)) this.emit('wasmRequestDetected', { url });

    const blocked = ['font', 'media', 'image', 'stylesheet'];
    if (blocked.includes(type)) return request.abort().catch(()=>{});
    request.continue().catch(()=>{});
  }

  async handleResponse(page, response) {
    const url = response.url();
    const headers = response.headers?.() || {};
    const contentType = (headers['content-type'] || '').toLowerCase();
    const length = parseInt(headers['content-length'] || '0', 10);

    try {
      if (length && length > this.config.maxResourceBytes) return;

      if (this.isWasmResponse(response, url)) {
        await this.downloadWasmResource(url, response);
      } else if (this.isJavaScriptContent(contentType)) {
        await this.downloadJavaScriptResource(url, response);
      } else if (this.isJsonContent(contentType)) {
        await this.downloadJsonResource(url, response);
      }
    } catch (e) {
      this.emit('resourceError', { url, error: e.message });
    }
  }

  // ========= processamento de página =========
  async processPageWithRetry(page, url, depth, referrer, isExternal, attempt = 1) {
    try {
      const resp = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: this.config.timeout });
      if (!resp || !resp.ok()) throw new Error(`HTTP ${resp?.status() || 'UNKNOWN'}: ${resp?.statusText() || 'Network Error'}`);

      await this.sleep(1500);
      const content = await page.content();
      await this.saveHtmlContent(url, content);

      await this.saveInlineScripts(page, url);

      // leve interação para disparar carregamentos dinâmicos
      await this.performLightInteraction(page);
      await this.sleep(2000);

      const maxDepthForPage = isExternal ? this.config.maxExternalDepth : this.config.maxDepth;
      if (depth < maxDepthForPage) await this.extractAndQueueLinks(page, url, depth, isExternal);

      this.emit('pageProcessed', { url, depth, isExternal, size: content.length });
    } catch (error) {
      if (attempt < this.config.retryAttempts) {
        await this.sleep(1000 * attempt);
        return this.processPageWithRetry(page, url, depth, referrer, isExternal, attempt + 1);
      }
      throw error;
    }
  }

  async performLightInteraction(page) {
    try {
      await page.evaluate(() => {
        ['load', 'DOMContentLoaded', 'click', 'focus'].forEach(ev =>
          document.dispatchEvent(new Event(ev))
        );
        const clickable = Array.from(document.querySelectorAll('button, [onclick], [role="button"]')).slice(0, 5);
        clickable.forEach(el => { try { el.click(); } catch {} });
      });
    } catch {}
  }

  async extractAndQueueLinks(page, currentUrl, depth, isCurrentExternal) {
    try {
      const links = await page.evaluate(() => {
        const a = Array.from(document.querySelectorAll('a[href]'));
        return a.slice(0, 50).map(x => ({ href: x.href, text: (x.textContent || '').trim().slice(0, 100) }));
      });

      for (const link of links) {
        try {
          const absoluteUrl = new URL(link.href, currentUrl).href;
          const parsed = new URL(absoluteUrl);
          const baseParsed = new URL(location.origin);

          const isLinkExternal = parsed.hostname !== baseParsed.hostname &&
            !(parsed.hostname.replace(/^www\./, '').endsWith('.' + baseParsed.hostname.replace(/^www\./, '')));

          if (isLinkExternal && isCurrentExternal) continue;

          if (isLinkExternal) {
            const domain = parsed.hostname;
            const cur = this.state.externalDomainCount.get(domain) || 0;
            if (cur >= this.config.maxExternalLinksPerDomain) continue;
            this.state.externalDomainCount.set(domain, cur + 1);
          }

          if (this.isUrlAllowed(parsed, isLinkExternal) &&
              !this.state.visitedUrls.has(absoluteUrl) &&
              !this.state.failedUrls.has(absoluteUrl)) {
            this.queue.push({
              url: absoluteUrl,
              depth: depth + 1,
              priority: this.calculatePriority(link.text, isLinkExternal),
              referrer: currentUrl,
              isExternal: isLinkExternal,
            });
          }
        } catch {}
      }
    } catch {}
  }

  // ========= salvamento / deduplicação =========
  async computeHash(buf) { return createHash('sha256').update(buf).digest('hex'); }

  async saveBufferUnique(buffer, dir, baseName, ext, meta = {}) {
    const hash = await this.computeHash(buffer);
    if (this.hashIndex.has(hash)) return { reused: true, ...this.hashIndex.get(hash), hash };
    const filename = `${baseName}_${hash.slice(0, 12)}.${ext}`;
    const filePath = path.join(this.baseDir, dir, filename);
    await fs.writeFile(filePath, buffer);
    const info = { path: filePath, size: buffer.length, ...meta };
    this.hashIndex.set(hash, info);
    this.manifest.push({ type: ext, sha256: hash, path: filePath, ...meta });
    return { reused: false, ...info, hash };
  }

  async saveTextUnique(text, dir, baseName, ext, meta = {}) {
    return this.saveBufferUnique(Buffer.from(text, 'utf8'), dir, baseName, ext, meta);
  }

  sanitizeUrlBase(u) {
    try {
      const { pathname } = new URL(u);
      const base = path.basename(pathname) || 'resource';
      return base.replace(/\.[^.]*$/, '').replace(/[^a-zA-Z0-9._-]/g, '_');
    } catch { return 'resource'; }
  }

  async saveHtmlContent(url, content) {
    try {
      const base = this.sanitizeUrlBase(url);
      await this.saveTextUnique(content, 'html', base, 'html', { timestamp: Date.now(), url, source: 'document' });
      this.emit('htmlSaved', { url, size: content.length });
    } catch {}
  }

  async saveInlineScripts(page, url) {
    try {
      const scripts = await page.$$eval('script:not([src])', nodes =>
        nodes.map((n, i) => ({ idx: i, type: n.type || 'text/javascript', text: n.textContent || '' }))
             .filter(s => (s.text || '').trim().length > 0)
      );
      for (const s of scripts) {
        const baseName = this.sanitizeUrlBase(url) + `_inline_${s.idx}`;
        const saved = await this.saveTextUnique(s.text, 'js', baseName, 'js',
          { timestamp: Date.now(), source: 'inline', url, mime: s.type });
        this.emit('resourceSaved', { type: 'js', url: `${url}#inline-${s.idx}`, path: saved.path });
      }
    } catch {}
  }

  async saveDynamicWasm(wasmData) {
    try {
      const { buffer, timestamp, url, context } = wasmData;
      const wasmBuffer = Buffer.from(buffer);
      if (wasmBuffer.length < 4 || wasmBuffer.readUInt32LE(0) !== 0x6d736100) return;
      const base = this.sanitizeUrlBase(url || 'dynamic');
      const meta = { timestamp: timestamp || Date.now(), source: 'dynamic', url, context: context || '' };
      const saved = await this.saveBufferUnique(wasmBuffer, 'wasm', base, 'wasm', meta);
      this.resources.wasm.set(`dynamic:${saved.hash}`, saved);
      this.emit('resourceSaved', { type: 'wasm', url: `dynamic:${saved.hash}`, path: saved.path });
    } catch {}
  }

  async saveWasmFromNetwork(url, buffer) {
    try {
      if (buffer.length < 4 || buffer.readUInt32LE(0) !== 0x6d736100) return;
      const base = this.sanitizeUrlBase(url);
      const saved = await this.saveBufferUnique(buffer, 'wasm', base, 'wasm',
        { timestamp: Date.now(), source: 'network', url });
      this.resources.wasm.set(url, saved);
      this.emit('resourceSaved', { type: 'wasm', url, path: saved.path });
    } catch {}
  }

  async downloadWasmResource(url, response) {
    try {
      const buffer = await response.buffer();
      if (buffer.length < 4 || buffer.readUInt32LE(0) !== 0x6d736100) return;
      const base = this.sanitizeUrlBase(url);
      const saved = await this.saveBufferUnique(buffer, 'wasm', base, 'wasm',
        { timestamp: Date.now(), source: 'response', url });
      this.resources.wasm.set(url, saved);
      this.emit('resourceSaved', { type: 'wasm', url, path: saved.path });
    } catch {}
  }

  async downloadJavaScriptResource(url, response) {
    try {
      const content = await response.text();
      const base = this.sanitizeUrlBase(url);
      const saved = await this.saveTextUnique(content, 'js', base, 'js',
        { timestamp: Date.now(), source: 'response', url, mime: 'application/javascript' });
      this.resources.js.set(url, saved);
      this.emit('resourceSaved', { type: 'js', url, path: saved.path });
    } catch {}
  }

  async downloadJsonResource(url, response) {
    try {
      const content = await response.text();
      const base = this.sanitizeUrlBase(url);
      const saved = await this.saveTextUnique(content, 'json', base, 'json',
        { timestamp: Date.now(), source: 'response', url, mime: 'application/json' });
      this.resources.json.set(url, saved);
      // opcional: this.emit('resourceSaved', { type: 'json', url, path: saved.path });
    } catch {}
  }

  // ========= util de tipos =========
  looksLikeWasmUrl(url) {
    const u = url.toLowerCase();
    return u.endsWith('.wasm') || u.includes('.wasm?') || u.includes('/wasm/') ||
           u.includes('wasm-') || u.includes('worker') || u.includes('crypto') ||
           u.includes('mining') || u.includes('module');
  }

  isWasmResponse(response, url) {
    const headers = response.headers?.() || {};
    const ct = (headers['content-type'] || '').toLowerCase();
    const mt = (response.mimeType || '').toLowerCase();
    const u = url.toLowerCase();
    return [
      ct.includes('application/wasm'),
      ct.includes('application/octet-stream'),
      mt.includes('application/wasm'),
      mt.includes('application/octet-stream'),
      u.endsWith('.wasm'),
      u.includes('.wasm?'),
      u.includes('/wasm/'),
      u.includes('wasm-'),
      (u.includes('worker') && ct.includes('octet-stream')),
      (u.includes('crypto') && ct.includes('octet-stream')),
      (u.includes('mining') && ct.includes('octet-stream')),
    ].some(Boolean);
  }

  isJavaScriptContent(contentType) {
    const ct = (contentType || '').toLowerCase();
    return ct.includes('javascript') || ct.includes('text/javascript') || ct.includes('application/x-javascript');
  }
  isJsonContent(contentType) { return (contentType || '').toLowerCase().includes('application/json'); }

  // ========= políticas de navegação =========
  isSameDomainOrSubdomain(parsedUrl, baseUrlParsed) {
    if (parsedUrl.hostname === baseUrlParsed.hostname) return true;
    if (!this.config.allowSubdomains) return false;
    const base = baseUrlParsed.hostname.replace(/^www\./, '');
    const target = parsedUrl.hostname.replace(/^www\./, '');
    return target === base || target.endsWith('.' + base);
  }

  isUrlAllowed(parsedUrl, isExternal) {
    if (this.config.blockedDomains.includes(parsedUrl.hostname)) return false;
    if (this.config.allowedDomains.length > 0) return this.config.allowedDomains.includes(parsedUrl.hostname);
    if (isExternal) return this.config.allowExternalDomains || this.config.maxExternalLinksPerDomain > 0;
    return this.isSameDomainOrSubdomain(parsedUrl, new URL(this.baseUrl));
  }

  calculatePriority(linkText, isExternal) {
    const base = isExternal ? 4 : 1;
    const t = (linkText || '').toLowerCase();
    if (['home', 'main', 'index', 'product', 'service'].some((s) => t.includes(s))) return base;
    if (['contact', 'about', 'privacy', 'terms'].some((s) => t.includes(s))) return base + 2;
    return base + 1;
  }

  // ========= relatório / finalização =========
  async generateReport(duration) {
    const report = {
      sessionId: this.sessionId,
      timestamp: new Date().toISOString(),
      durationMs: duration,
      statistics: {
        pagesProcessed: this.state.crawledCount,
        pagesFailed: this.state.failedUrls.size,
        resources: {
          wasm: this.resources.wasm.size,
          js: this.resources.js.size,
          json: this.resources.json.size,
        },
      },
      failedUrls: Array.from(this.state.failedUrls),
      config: this.config,
    };
    const manifestPath = path.join(this.baseDir, 'manifest.jsonl');
    await fs.writeFile(manifestPath, this.manifest.map((x) => JSON.stringify(x)).join('\n'));
    report.manifest = path.basename(manifestPath);

    const reportPath = path.join(this.baseDir, 'crawl_report.json');
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
  }

  getTotalResourceCount() {
    return this.resources.wasm.size + this.resources.js.size + this.resources.json.size;
  }

  async cleanup() { try { if (this.browser) await this.browser.close(); } catch {} }
  sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
}

// ========= PriorityQueue =========
class PriorityQueue {
  constructor({ comparator }) { this.heap = []; this.comparator = comparator; }
  push(item) { this.heap.push(item); this._up(this.heap.length - 1); }
  pop() {
    if (this.heap.length === 0) return undefined;
    if (this.heap.length === 1) return this.heap.pop();
    const root = this.heap[0];
    this.heap[0] = this.heap.pop(); this._down(0);
    return root;
  }
  isEmpty() { return this.heap.length === 0; }
  size() { return this.heap.length; }
  _up(i) {
    while (i > 0) {
      const p = Math.floor((i - 1) / 2);
      if (this.comparator(this.heap[i], this.heap[p]) >= 0) break;
      [this.heap[i], this.heap[p]] = [this.heap[p], this.heap[i]];
      i = p;
    }
  }
  _down(i) {
    while (true) {
      const l = 2 * i + 1, r = 2 * i + 2;
      let s = i;
      if (l < this.heap.length && this.comparator(this.heap[l], this.heap[s]) < 0) s = l;
      if (r < this.heap.length && this.comparator(this.heap[r], this.heap[s]) < 0) s = r;
      if (s === i) break;
      [this.heap[i], this.heap[s]] = [this.heap[s], this.heap[i]];
      i = s;
    }
  }
}

// ========= CLI =========
async function main() {
  const targetUrl = process.argv[2];
  if (!targetUrl) {
    console.error('Uso: node crawler.js <URL>');
    console.error('Opções: --external --external-limit=N --no-subs --allow=a,b --block=a,b');
    console.error('         --pages=N --depth=N --concurrent=N --max-bytes=N --headless');
    process.exit(1);
  }

  const args = process.argv.slice(3);
  const parseArg = (prefix, def) => {
    const a = args.find((x) => x.startsWith(prefix));
    if (!a) return def;
    const v = a.split('=')[1];
    const n = Number(v);
    return Number.isFinite(n) ? n : def;
  };
  const parseListArg = (prefix) => {
    const a = args.find((x) => x.startsWith(prefix));
    return a ? a.split('=')[1].split(',').map((s) => s.trim()).filter(Boolean) : [];
  };

  const config = {
    allowExternalDomains: args.includes('--external'),
    allowSubdomains: !args.includes('--no-subs'),
    headless: args.includes('--headless') ? 'new' : 'new',
    maxPages: parseArg('--pages=', 50),
    maxDepth: parseArg('--depth=', 3),
    concurrentPages: parseArg('--concurrent=', 2),
    maxExternalLinksPerDomain: parseArg('--external-limit=', 3),
    maxResourceBytes: parseArg('--max-bytes=', 25 * 1024 * 1024),
    allowedDomains: parseListArg('--allow='),
    blockedDomains: parseListArg('--block='),
  };

  console.log('Advanced Web Crawler (v40)\n==========================');
  console.log(`Target: ${targetUrl}`);
  console.log(`Pages: ${config.maxPages}, Depth: ${config.maxDepth}, Concurrent: ${config.concurrentPages}`);
  console.log('');

  const crawler = new AdvancedCrawler(config);

  crawler.on('initialized', ({ targetUrl, baseDir }) => {
    console.log(`Initialized for: ${targetUrl}`);
    console.log(`Output directory: ${baseDir}\n`);
  });
  crawler.on('progress', ({ crawled, queued, active }) => {
    console.log(`Progress: ${crawled} processed, ${queued} queued, ${active} active`);
  });
  crawler.on('pageProcessed', ({ url, depth, isExternal, size }) => {
    const type = isExternal ? 'EXT' : 'INT';
    const shortUrl = url.length > 80 ? url.slice(0, 77) + '...' : url;
    console.log(`[${type}] ${shortUrl} (depth: ${depth}, size: ${Math.round(size / 1024)}KB)`);
  });
  crawler.on('resourceSaved', ({ type, url, path: p }) => {
    const filename = p.split(path.sep).pop();
    if (type === 'wasm') console.log(`  WASM saved: ${filename}`);
  });
  crawler.on('wasmRequestDetected', ({ url }) => console.log(`  WASM request detected: ${url}`));
  crawler.on('pageError', ({ url, error }) => {
    const shortUrl = url.length > 60 ? url.slice(0, 57) + '...' : url;
    console.error(`Error: ${shortUrl} - ${error}`);
  });
  crawler.on('crawlCompleted', ({ duration, pagesProcessed, resourcesFound }) => {
    console.log('\nCrawling Completed\n==================');
    console.log(`Duration: ${duration} ms`);
    console.log(`Pages processed: ${pagesProcessed}`);
    console.log(`Resources found: ${resourcesFound}`);
    console.log(`  WASM files: ${crawler.resources.wasm.size}`);
    console.log(`  JS files: ${crawler.resources.js.size}`);
    console.log(`  JSON files: ${crawler.resources.json.size}`);
  });
  crawler.on('error', ({ stage, error }) => console.error(`Error in ${stage}:`, error.message));

  try {
    await crawler.init(targetUrl);
    await crawler.crawl();
  } catch (err) {
    console.error('Fatal error:', err.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch((e) => {
    console.error('Unhandled error:', e);
    process.exit(1);
  });
}

module.exports = { AdvancedCrawler, PriorityQueue };
