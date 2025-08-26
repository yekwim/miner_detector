# Miner Detector

Pipeline de coleta, normalização, embeddings e classificação para detectar mineração de criptomoedas em páginas web.

## Estrutura

- `crawler/` → Crawler em Node.js (Puppeteer) que coleta HTML, JS, Workers, WASM.
- `parser/` → Parser em Node.js que normaliza código, gera tokens e chunks (300–800 tokens).
- `scripts/`
  - `embed_runner.py` → Gera embeddings com Qwen3-Embedding-0.6B.
  - `train.py` → Treina classificador (Logistic Regression/SVM + calibração).
  - `predict.py` → Classificação em lote usando modelo treinado.
- `data/` → Diretórios de dados.
- `models/` → Modelos treinados.
- `env/requirements.txt` → Dependências Python.

## Fluxo

### Inferência (produção)
1. `crawler/crawler.js` → gera `data/crawl_output/`.
2. `parser/parser_v2.js` → gera `data/parser_output/chunks/embeddings_chunks.jsonl`.
3. `scripts/embed_runner.py` → gera `data/embeddings/`.
4. `scripts/predict.py` → usa `models/latest/model.joblib` para pontuar.

### Treinamento (offline)
1. Dataset rotulado em `data/labels/labels.csv`.
2. `scripts/embed_runner.py` em `data/parser_output/` rotulado.
3. `scripts/train.py` → salva modelo calibrado em `models/latest/`.

## Labels

Formato esperado em `data/labels/labels.csv`:
