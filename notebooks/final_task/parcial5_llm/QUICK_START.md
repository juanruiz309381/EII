# Quick Start Guide - Parcial 5 LLM

Guía rápida para ejecutar el proyecto en 5 minutos.

## Prerequisitos

```bash
# Verificar Python
python3 --version  # Debe ser >= 3.8

# Verificar Ollama
ollama --version  # Si no está instalado, ver instalación abajo
```

## Instalación Rápida de Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Verificar
ollama --version
```

## Descargar Modelos (10-15 minutos)

```bash
# Descargar los 3 modelos
ollama pull llama3.2:1b
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5:0.5b

# Verificar
ollama list
```

## Opción 1: Jupyter Notebook (Recomendado)

```bash
# Instalar dependencias
pip install jupyter pandas matplotlib seaborn

# Navegar al notebook
cd parcial5_llm/notebooks

# Abrir Jupyter
jupyter notebook LLM_Evaluation.ipynb
```

**En el notebook:**
1. Ejecutar todas las celdas: `Kernel > Restart & Run All`
2. Esperar 15-20 minutos (pruebas automáticas)
3. Ver resultados y gráficos generados
4. Revisar archivos en `../results/`

## Opción 2: Script Python

```bash
# Navegar a scripts
cd parcial5_llm/scripts

# Ejecutar evaluación
python3 test_llm_models.py
```

Resultados en: `parcial5_llm/results/`

## Opción 3: Open WebUI (Manual)

```bash
# Con Docker
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

# Abrir navegador
# http://localhost:3000
```

Interactuar manualmente con cada modelo.

## Probar Rápidamente un Modelo

```bash
# Chat interactivo
ollama run llama3.2:1b

# Pregunta de prueba
> Hola, ¿cómo estás?

# Salir
> /bye
```

## Estructura de Archivos Generados

```
parcial5_llm/
├── results/
│   ├── evaluation_results.json      # Resultados completos
│   ├── comparison_report.md         # Reporte comparativo
│   ├── detailed_results.csv         # Datos tabulados
│   └── best_model_conversation.json # Conversación extendida
└── docs/
    └── informe_tecnico.md           # Informe final
```

## Troubleshooting Rápido

**Ollama no responde:**
```bash
ollama serve
```

**Modelo no encontrado:**
```bash
ollama pull llama3.2:1b
```

**Puerto 3000 ocupado (Open WebUI):**
```bash
docker run -d -p 3001:8080 ...  # Cambiar 3000 por 3001
```

## Siguiente Paso

Lee el README.md completo para detalles técnicos y el informe_tecnico.md para análisis profundo.

## Contacto

Preguntas: [tu_email@ejemplo.com]
