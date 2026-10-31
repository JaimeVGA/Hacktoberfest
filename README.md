# RAG jurídico de Oaxaca

Pipeline local para consultar leyes de Oaxaca con recuperación aumentada (RAG),
ChromaDB, grafo de relaciones y Ollama.

## Tecnologías

- Python 3
- LangChain
- PyPDFLoader
- ChromaDB persistente
- Ollama
- `nomic-embed-text` para embeddings
- `qwen2.5:1.5b` para respuestas ligeras
- Tesseract OCR y ImageMagick para PDFs escaneados
- `pdftoppm` para rasterizar páginas

## Uso

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
python Conversion.py ingest
python Conversion.py query "¿Qué establece el artículo 2?"
```

`ingest` detecta todos los PDFs del directorio, extrae unidades por artículo,
conserva capítulo, página y fuente, genera `ley_grafo.json` y reconstruye
`chroma_db/`.

## OCR de PDFs escaneados

Cuando un PDF contiene menos de 1000 caracteres extraíbles, se activa OCR:

- Resolución: `300 DPI`
- Idiomas: `spa+eng`
- Segmentación Tesseract: `--psm 3`
- Preprocesamiento: escala de grises, deskew, contraste y enfoque
- Caché: `9.ocr.v2.txt`
- Modelo español: `tessdata/spa.traineddata`

Parámetros opcionales:

```bash
OCR_DPI=300 OCR_PSM=3 OCR_LANG=spa+eng python Conversion.py ingest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text \
OLLAMA_LLM_MODEL=qwen2.5:1.5b \
python Conversion.py query "pregunta"
```

## Archivos generados

- `chroma_db/`: índice vectorial local
- `ley_grafo.json`: nodos de capítulos/artículos y relaciones
- `9.ocr.v2.txt`: texto OCR cacheado de `9.pdf`

## Motor GraphRAG con Groq (`rag_engine.py`)

Consulta el grafo (`ley_grafo.json`, con el texto de cada artículo) con búsqueda difusa,
expansión a 1–2 saltos y respuesta con Groq (modelo abierto `openai/gpt-oss-120b`).

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # completar GROQ_API_KEY y, opcionalmente, GROQ_MODEL
python export_texts.py        # regenera ley_grafo.json desde chroma_db tras cada `ingest`
python test_cli.py --debug    # CLI interactivo
```

Para el bot: `from rag_engine import ask_rag` → `ask_rag(pregunta, history=None) -> str`
(síncrona; en un bot asíncrono usar `asyncio.to_thread`). No lanza excepciones.
