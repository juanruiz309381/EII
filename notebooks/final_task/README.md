# Parcial 5: Implementación y Evaluación de Modelos LLM

## Guía Completa Paso a Paso

**Autor:** [Tu Nombre]
**Fecha:** Noviembre 2025
**Curso:** Ingeniería de Información

---

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Requisitos Previos](#requisitos-previos)
3. [Instalación de Ollama](#instalación-de-ollama)
4. [Instalación de Open WebUI](#instalación-de-open-webui)
5. [Descarga de Modelos](#descarga-de-modelos)
6. [Ejecución de Pruebas](#ejecución-de-pruebas)
7. [Análisis de Resultados](#análisis-de-resultados)
8. [Estructura del Proyecto](#estructura-del-proyecto)
9. [Referencias](#referencias)

---

## Introducción

Este proyecto implementa y evalúa tres modelos de lenguaje (LLM) livianos usando Ollama, cumpliendo con los requisitos del Parcial 5:

1. ✅ Implementación de al menos 3 modelos LLM
2. ✅ Pruebas conversacionales
3. ✅ Informe de desempeño
4. ✅ Conversación coherente con el mejor modelo
5. ✅ Informe técnico
6. ✅ Video explicativo

### Modelos Seleccionados

| Modelo | Tamaño | Características |
|--------|--------|----------------|
| **llama3.2:1b** | 1.3 GB | Meta's Llama 3.2, rápido y eficiente |
| **deepseek-r1:1.5b** | ~1.1 GB | DeepSeek R1, especializado en razonamiento |
| **qwen2.5:0.5b** | ~397 MB | Qwen 2.5, ultraligero y rápido |

---

## Requisitos Previos

### Hardware Mínimo
- **RAM:** 8 GB (recomendado 16 GB)
- **Almacenamiento:** 10 GB libres
- **CPU:** Multi-core (mínimo 4 cores)
- **GPU:** Opcional (acelera inferencia)

### Software
- **Sistema Operativo:** Linux (Ubuntu 20.04+), macOS, o Windows con WSL2
- **Python:** 3.8 o superior
- **Git:** Para clonar repositorios

---

## Instalación de Ollama

### Linux / macOS

```bash
# Método 1: Script oficial
curl -fsSL https://ollama.com/install.sh | sh

# Método 2: Instalación manual
# Descargar desde: https://ollama.com/download
```

### Verificar Instalación

```bash
ollama --version
# Debe mostrar: ollama version is 0.x.x
```

### Iniciar Servicio Ollama

```bash
# El servicio debería iniciarse automáticamente
# Si no, ejecutar:
ollama serve
```

---

## Instalación de Open WebUI

Open WebUI proporciona una interfaz web moderna para interactuar con modelos de Ollama.

### Opción 1: Docker (Recomendado)

```bash
# Instalar Docker si no lo tienes
curl -fsSL https://get.docker.com | sh

# Ejecutar Open WebUI
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

### Opción 2: Instalación Local con Python

```bash
# Clonar repositorio
git clone https://github.com/open-webui/open-webui.git
cd open-webui

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python -m open_webui
```

### Acceder a Open WebUI

1. Abrir navegador en: `http://localhost:3000`
2. Crear cuenta (primera vez)
3. Los modelos de Ollama aparecerán automáticamente

---

## Descarga de Modelos

### Descargar los 3 Modelos Principales

```bash
# Modelo 1: Llama 3.2 1B
ollama pull llama3.2:1b

# Modelo 2: DeepSeek R1 1.5B
ollama pull deepseek-r1:1.5b

# Modelo 3: Qwen 2.5 0.5B
ollama pull qwen2.5:0.5b
```

### Verificar Modelos Instalados

```bash
ollama list
```

Deberías ver algo como:

```
NAME                ID              SIZE      MODIFIED
llama3.2:1b         baf6a787fdff    1.3 GB    2 minutes ago
deepseek-r1:1.5b    aabd4debf0c8    1.1 GB    5 minutes ago
qwen2.5:0.5b        c5396e06af29    397 MB    8 minutes ago
```

### Prueba Rápida de un Modelo

```bash
# Probar llama3.2
ollama run llama3.2:1b "Hola, ¿cómo estás?"

# Salir de la sesión interactiva
/bye
```

---

## Ejecución de Pruebas

### Estructura del Proyecto

```
parcial5_llm/
├── notebooks/
│   └── LLM_Evaluation.ipynb    # Notebook principal de evaluación
├── scripts/
│   └── test_llm_models.py      # Script automatizado de pruebas
├── results/
│   ├── evaluation_results.json
│   ├── comparison_report.md
│   ├── detailed_results.csv
│   └── best_model_conversation.json
├── docs/
│   └── informe_tecnico.md
└── assets/
    └── screenshots/
```

### Método 1: Usando Jupyter Notebook (Recomendado)

```bash
# Instalar dependencias
pip install jupyter pandas matplotlib seaborn

# Navegar al directorio
cd parcial5_llm/notebooks

# Iniciar Jupyter
jupyter notebook LLM_Evaluation.ipynb
```

**Pasos en el Notebook:**
1. Ejecutar celda de configuración inicial
2. Verificar modelos disponibles
3. Ejecutar pruebas conversacionales
4. Analizar resultados con gráficos
5. Realizar conversación con mejor modelo
6. Exportar resultados

### Método 2: Script Python Automatizado

```bash
cd parcial5_llm/scripts

# Ejecutar script de evaluación
python3 test_llm_models.py
```

Este script:
- Prueba los 3 modelos con 8 categorías de preguntas
- Mide tiempos de respuesta
- Evalúa calidad de respuestas
- Genera reportes JSON y Markdown

### Método 3: Open WebUI (Interfaz Gráfica)

1. Abrir `http://localhost:3000`
2. Seleccionar modelo en el menú desplegable
3. Escribir preguntas y evaluar respuestas manualmente
4. Tomar capturas de pantalla para el informe

---

## Análisis de Resultados

### Métricas Evaluadas

1. **Tiempo de Respuesta**
   - Promedio, mediana, mínimo, máximo
   - Por categoría de pregunta

2. **Calidad de Respuestas**
   - Coherencia
   - Relevancia (keywords matching)
   - Longitud de respuesta

3. **Tasa de Éxito**
   - Porcentaje de respuestas sin errores
   - Timeouts

### Visualizaciones Generadas

- Gráficos de barras: Tiempo promedio por modelo
- Boxplots: Distribución de tiempos
- Gráficos comparativos por categoría
- Heatmaps de desempeño

### Interpretar Resultados

Los resultados se guardan en `results/`:

```bash
# Ver reporte comparativo
cat results/comparison_report.md

# Ver datos detallados
cat results/evaluation_results.json

# Abrir CSV en Excel/LibreOffice
libreoffice results/detailed_results.csv
```

---

## Estructura del Proyecto

### Archivos Principales

#### `notebooks/LLM_Evaluation.ipynb`
Notebook interactivo con:
- Configuración y pruebas
- Visualizaciones
- Análisis estadístico
- Exportación de resultados

#### `scripts/test_llm_models.py`
Script automatizado que:
- Define clase `LLMTester`
- Ejecuta pruebas sistemáticas
- Genera reportes automáticos

#### `results/`
Directorio con outputs:
- `evaluation_results.json`: Resultados completos
- `comparison_report.md`: Reporte comparativo
- `detailed_results.csv`: Datos tabulados
- `best_model_conversation.json`: Conversación extendida

#### `docs/informe_tecnico.md`
Informe técnico completo con:
- Introducción y objetivos
- Metodología
- Resultados y análisis
- Conclusiones
- Referencias

---

## Ejecución Completa: Checklist

### Preparación
- [ ] Instalar Ollama
- [ ] Instalar Open WebUI (opcional)
- [ ] Descargar los 3 modelos
- [ ] Instalar Python y dependencias
- [ ] Clonar/crear estructura de proyecto

### Pruebas
- [ ] Ejecutar prueba rápida con cada modelo
- [ ] Correr notebook completo o script Python
- [ ] Verificar generación de archivos en `results/`
- [ ] Realizar conversación extendida con mejor modelo

### Documentación
- [ ] Completar informe técnico
- [ ] Tomar capturas de pantalla
- [ ] Generar gráficos y visualizaciones
- [ ] Revisar y validar resultados

### Entrega
- [ ] Informe técnico en PDF/Markdown
- [ ] Notebooks con outputs visibles
- [ ] Archivos de resultados
- [ ] Video explicativo (ver script en `docs/`)

---

## Solución de Problemas Comunes

### Problema: "ollama: command not found"
```bash
# Verificar instalación
which ollama

# Reinstalar si es necesario
curl -fsSL https://ollama.com/install.sh | sh
```

### Problema: "Model not found"
```bash
# Listar modelos disponibles
ollama list

# Volver a descargar
ollama pull llama3.2:1b
```

### Problema: Timeout en respuestas
- Aumentar timeout en el código
- Verificar recursos del sistema (RAM/CPU)
- Cerrar aplicaciones pesadas

### Problema: Error de conexión con Ollama
```bash
# Verificar que el servicio esté corriendo
ps aux | grep ollama

# Reiniciar servicio
ollama serve
```

---

## Comandos Útiles

### Gestión de Modelos

```bash
# Listar modelos
ollama list

# Eliminar modelo
ollama rm modelo:tag

# Ver información del modelo
ollama show llama3.2:1b

# Actualizar modelo
ollama pull llama3.2:1b
```

### Uso Interactivo

```bash
# Iniciar chat interactivo
ollama run llama3.2:1b

# Con parámetros personalizados
ollama run llama3.2:1b --temperature 0.8 --top-p 0.9

# Comandos dentro del chat:
# /bye         - Salir
# /clear       - Limpiar contexto
# /set         - Ver/cambiar parámetros
```

### API REST

```bash
# Consulta vía API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:1b",
  "prompt": "¿Qué es la inteligencia artificial?",
  "stream": false
}'
```

---

## Tips y Mejores Prácticas

1. **Optimización de Rendimiento**
   - Usar modelos más pequeños para pruebas rápidas
   - Cerrar aplicaciones innecesarias
   - Monitorear uso de RAM con `htop`

2. **Diseño de Prompts**
   - Ser específico y claro
   - Incluir contexto relevante
   - Usar ejemplos cuando sea necesario

3. **Evaluación de Modelos**
   - Probar con diversos tipos de tareas
   - Repetir pruebas para validar consistencia
   - Documentar hallazgos anómalos

4. **Gestión de Recursos**
   - Los modelos se cachean en `~/.ollama/models`
   - Limpiar modelos no usados para liberar espacio
   - Usar GPU si está disponible (detección automática)

---

## Referencias

### Documentación Oficial
- **Ollama:** https://ollama.com/
- **Open WebUI:** https://docs.openwebui.com/
- **Llama 3.2:** https://llama.meta.com/
- **DeepSeek:** https://www.deepseek.com/
- **Qwen:** https://qwenlm.github.io/

### Tutoriales y Recursos
- Ollama GitHub: https://github.com/ollama/ollama
- Open WebUI GitHub: https://github.com/open-webui/open-webui
- Awesome Ollama: https://github.com/jmorganca/awesome-ollama

### Papers y Artículos
- "Llama: Open Foundation and Fine-Tuned Chat Models" (Meta AI, 2024)
- "Qwen Technical Report" (Alibaba, 2024)
- "The Illustrated Transformer" (Jay Alammar)

---

## Contacto y Soporte

Para preguntas sobre este proyecto:
- **Email:** [tu_email@ejemplo.com]
- **GitHub:** [tu_usuario]

Para issues con las herramientas:
- Ollama Issues: https://github.com/ollama/ollama/issues
- Open WebUI Issues: https://github.com/open-webui/open-webui/issues

---

## Licencia

Este proyecto se desarrolla con fines educativos para el Parcial 5 del curso de Ingeniería de Información.

Los modelos utilizados tienen sus propias licencias:
- Llama 3.2: Llama 3 Community License
- DeepSeek: MIT License
- Qwen: Apache 2.0 License

---

**¡Buena suerte con tu evaluación! 🚀**
