# 🚦 Asistente Legal Vial de Oaxaca de Juárez (GraphRAG + Telegram Bot)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework: Open Source / Open Weights](https://img.shields.io/badge/Models-Open--Weights-green.svg)](https://groq.com)

Un asistente jurídico de código abierto y libre acceso para orientar a la ciudadanía sobre tránsito, vialidad, multas, grúas y movilidad en el **Municipio de Oaxaca de Juárez** y el Estado de Oaxaca.

Combina técnicas de **GraphRAG** (recuperación aumentada mediante grafos de conocimiento jurídico), búsqueda léxico-difusa optimizada para jerga popular mexicana, modelos abiertos de lenguaje (`openai/gpt-oss-120b` en Groq), transcripción de notas de voz con **Whisper** y un **Bot interactivo de Telegram**.

---

## 🏛️ Normativas y Documentos Indexados

El sistema indexa de forma estructurada (por capítulo y artículo) la normativa vigente, priorizando el ámbito municipal:

1. **Reglamento de Tránsito Metropolitano del Municipio de Oaxaca de Juárez** (*Ámbito Municipal - Prioritario*).
2. **Ley de Movilidad para el Estado de Oaxaca** (*Ámbito Estatal*).
3. **Ley de Tránsito y Vialidad del Estado de Oaxaca** (*Ámbito Estatal*).
4. **Ley de Ingresos del Estado de Oaxaca para el Ejercicio Fiscal 2026** (*Ámbito Estatal*).
5. **Iniciativas y Decretos de Reforma Vial** (documentos auxiliares digitalizados con OCR de alta resolución).

---

## ✨ Características Principales

- **🧠 Motor GraphRAG:** Grafo de conocimiento legal que enlaza capítulos, artículos secuenciales y fuentes normativas, permitiendo recuperación por saltos topológicos (1-2 saltos).
- **🗣️ Tolerancia a Jerga y Coloquialismos:** Diccionario integrado que mapea términos de uso común (*«troca»*, *«remolcaron»*, *«corralón»*, *«mordida»*, *«borracho»*, *«doble fila»*) a los preceptos formales de las leyes (*vehículo, arrastre, depósito, cohecho, alcoholimetría*).
- **🎙️ Soporte para Notas de Voz:** Permite a los usuarios enviar mensajes de voz en Telegram; el bot transcribe el audio con `whisper-large-v3-turbo` y responde de inmediato.
- **⚡ Caché LRU en Memoria:** Respuestas ultrarrápidas (<5ms) para consultas frecuentes sin consumir cuota de tokens.
- **📜 Visor de Artículos Oficiales:** Botones interactivos en Telegram para desplegar el texto íntegro y literal del artículo citado sin costo computacional adicional.
- **🎯 Respuestas Concisas y Fundamentadas:** Respuestas directas al punto (sin redundancia ni rodeos) acompañadas de su respectivo sustento normativo y aviso legal.
- **🛡️ 100% de Código Abierto / Pesos Abiertos:** Utiliza modelos abiertos (`gpt-oss-120b`, `whisper-large-v3-turbo`, `nomic-embed-text`) y librerías FOSS (Fast Open Source Software).

---

## 🛠️ Arquitectura del Sistema

```mermaid
flowchart TD
    subgraph Ingesta [1. Pipeline de Ingestión]
        PDFs[PDFs Leyes y Reglamentos] --> OCR[OCR Tesseract 300 DPI + Deskew]
        OCR --> Chunker[Segmentador por Artículo/Capítulo]
        Chunker --> Chroma[(ChromaDB Embeddings)]
        Chunker --> GraphFile[ley_grafo.json\nGrafo de Artículos y Texto]
    end

    subgraph Core [2. Motor GraphRAG]
        Query[Consulta Texto o Voz] --> VoiceProc{¿Es Nota de Voz?}
        VoiceProc -- Sí --> Whisper[Groq Whisper Turbo]
        VoiceProc -- No --> CacheCheck{¿En Caché LRU?}
        Whisper --> CacheCheck
        CacheCheck -- Sí (Hit) --> RetAnswer[Retorno Inmediato]
        CacheCheck -- No (Miss) --> Fuzzy[Búsqueda Difusa + Jerga]
        Fuzzy --> GraphBFS[Expansión en Grafo 1-2 saltos]
        GraphBFS --> Prompt[Prompt Jurídico Conciso]
        Prompt --> LLM[Groq: gpt-oss-120b]
        LLM --> Formatter[Formato + Cita + Aviso Legal]
    end

    subgraph UI [3. Interfaces]
        Formatter --> CLI[test_cli.py]
        Formatter --> Telegram[telegram_bot.py]
    end
```

---

## 🚀 Instalación y Puesta en Marcha

### Prerrequisitos
- Python 3.10 o superior.
- Git.
- Opcional: Tesseract OCR y `poppler-utils` (solo si se van a reindexar PDFs escaneados desde cero).

### 1. Clonar el repositorio
```bash
git clone https://github.com/JaimeVGA/Hacktoberfest.git
cd Hacktoberfest
```

### 2. Crear y activar entorno virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
Copia la plantilla de configuración:
```bash
cp .env.example .env
```
Edita `.env` con tus credenciales:
```ini
GROQ_API_KEY=gsk_tu_clave_de_groq
TELEGRAM_BOT_TOKEN=tu_token_de_botfather
GROQ_MODEL=openai/gpt-oss-120b
```

---

## 💻 Uso

### A. Bot de Telegram (Recomendado)
Inicia el bot interactivo:
```bash
python telegram_bot.py
```
- Escribe a tu bot en Telegram o pulsa `/start`.
- Puedes seleccionar preguntas frecuentes de los botones, escribir consultas o enviar **notas de voz**.
- Consulta artículos completos pulsando los botones al pie de cada respuesta.

### B. Consola Interactiva (CLI)
Para probar consultas locales sin Telegram:
```bash
python test_cli.py --debug
```
*Comandos disponibles en el CLI: `/hops 1|2`, `/k N`, `/debug`, `/salir`.*

### C. Reindexar Documentos (Opcional)
Si agregas nuevos reglamentos o PDFs a la carpeta:
```bash
python Conversion.py ingest
```
*Esto reconstruirá tanto `chroma_db/` como `ley_grafo.json` con los textos completos y relaciones automáticamente.*

---

## 🧪 Pruebas Automatizadas

El proyecto cuenta con una suite completa de pruebas unitarias que validan la normalización, la búsqueda difusa, la tolerancia a jerga, la expansión del grafo, la caché LRU, el mock de audio y los manejadores de Telegram:

```bash
python -m pytest -v
```

---

## ⚖️ Exención de Responsabilidad Legal (Disclaimer)

Este software es una herramienta informativa y de divulgación ciudadana. Las respuestas generadas se fundamentan en el texto digitalizado de las leyes y reglamentos, pero no constituyen asesoramiento jurídico formal. Para casos específicos y controversias legales, consulte a un profesional del derecho o a las dependencias gubernamentales competentes.

---

## 📄 Licencia

Este proyecto está distribuido bajo la licencia libre y de código abierto **MIT License**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
