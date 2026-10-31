# Notas personales: construcción del RAG

## Qué se hizo

Se convirtió el contenido de tres PDFs legales en una base consultable:

1. Ley de Tránsito y Vialidad.
2. Ley de Movilidad.
3. `9.pdf`, un documento escaneado.

El corte principal no usa bloques arbitrarios de caracteres. Cada unidad
corresponde a un artículo y guarda sus metadatos: capítulo, número de artículo,
página y PDF de origen. Esto permite respuestas con referencias más precisas.

## Cómo funciona

`Conversion.py` realiza dos operaciones:

- `ingest`: descubre todos los PDFs, extrae artículos, genera el grafo y crea
  el índice Chroma.
- `query`: recupera los artículos más cercanos y los envía a Ollama con una
  instrucción para responder únicamente con el contexto legal.

El grafo se guarda en `ley_grafo.json`. Cada capítulo contiene artículos y cada
artículo se conecta con el siguiente. El identificador incluye el nombre del
PDF para evitar colisiones entre leyes.

## Tratamiento de `9.pdf`

El PDF no tenía texto digital suficiente. Por eso el programa detecta esa
condición y ejecuta:

1. `pdftoppm` para convertir cada página en imagen a 300 DPI.
2. ImageMagick para escala de grises, corrección de inclinación, contraste y
   enfoque.
3. Tesseract con `spa+eng` y `--psm 3`.
4. Caché en `9.ocr.v2.txt`, para no repetir el OCR en cada indexación.

El modelo español se encuentra en `tessdata/spa.traineddata`. Sin ese archivo,
el programa usa inglés como respaldo y muestra una advertencia.

## Modelos y parámetros

| Componente | Valor |
|---|---|
| Embeddings | `nomic-embed-text` |
| Modelo generativo | `qwen2.5:1.5b` |
| Recuperación | `k=4` documentos |
| OCR | `300 DPI`, `spa+eng`, `--psm 3` |
| Base vectorial | ChromaDB persistente |
| Grafo | JSON portable |

Se pueden cambiar los modelos sin editar el código:

```bash
OLLAMA_EMBEDDING_MODEL=nomic-embed-text \
OLLAMA_LLM_MODEL=qwen2.5:1.5b \
python Conversion.py query "pregunta"
```

## Reindexar

Después de agregar o reemplazar PDFs:

```bash
python Conversion.py ingest
```

El índice se elimina y se reconstruye para evitar documentos obsoletos.
