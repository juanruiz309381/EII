# Reconocimiento de Rangos de Edades por Voz

## Parcial 4 - Inteligencia Artificial II

**Instituto Tecnológico Metropolitano (ITM)**

---

## Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Instalación](#instalación)
4. [Uso](#uso)
5. [Metodología](#metodología)
6. [Resultados](#resultados)
7. [Documentación](#documentación)

---

## Descripción del Proyecto

Este proyecto implementa un sistema de **reconocimiento de rangos de edades** basado en audio de voz utilizando técnicas de Machine Learning convencional y Deep Learning con representación visual de datos mediante espectrogramas.

### Objetivos

- Clasificar voces en 4 categorías de edad: **Adolescente**, **Juvenil**, **Adulto**, **Adulto Mayor**
- Implementar modelos convencionales (Random Forest, SVM, XGBoost)
- Implementar modelos de Deep Learning con Transfer Learning (VGG16, ResNet50, MobileNetV2)
- Aplicar técnicas anti-overfitting (Dropout, Batch Normalization, Data Augmentation, Early Stopping)
- Generar métricas y visualizaciones completas

### Categorías de Edad

| Categoría | Rango de Edad | Origen en Dataset |
|-----------|---------------|-------------------|
| Adolescente | 13-19 años | teens |
| Juvenil | 20-29 años | twenties |
| Adulto | 30-49 años | thirties + fourties |
| Adulto Mayor | 50+ años | fifties + sixties |

---

## Estructura del Proyecto

```
exam4/
├── config/
│   └── config.yaml                 # Configuración del proyecto
├── src/
│   ├── preprocessing/
│   │   ├── audio_loader.py        # Carga de audios
│   │   ├── feature_extraction.py  # Extracción de características
│   │   └── data_augmentation.py   # Aumentación de datos
│   ├── models/
│   │   ├── conventional.py        # Modelos ML convencionales
│   │   └── deep_learning.py       # Modelos Deep Learning
│   └── utils/
│       ├── metrics.py             # Cálculo de métricas
│       └── visualization.py       # Visualizaciones
├── docs/
│   ├── INFORME_TECNICO.md         # Informe técnico completo
│   ├── METODOLOGIA.md             # Metodología detallada
│   ├── RESULTADOS.md              # Análisis de resultados
│   └── GUIA_USUARIO.md            # Guía de usuario
├── audios/
│   ├── organized_by_age/          # Audios organizados por categoría
│   └── cv-corpus-23.0-2025-09-05/ # Dataset original
├── results/
│   ├── models/                    # Modelos entrenados
│   ├── plots/                     # Gráficos y visualizaciones
│   └── metrics/                   # Reportes de métricas
├── train.py                       # Script principal de entrenamiento
├── requirements.txt               # Dependencias
└── README.md                      # Este archivo
```

---

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- GPU con CUDA (opcional, para Deep Learning)

### Pasos de Instalación

1. **Clonar o descargar el proyecto**

```bash
cd /ruta/al/proyecto/exam4
```

2. **Crear entorno virtual (recomendado)**

```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate  # En Windows
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

### Dependencias Principales

- `librosa`: Procesamiento de audio
- `tensorflow/keras`: Deep Learning
- `scikit-learn`: Machine Learning convencional
- `matplotlib/seaborn`: Visualizaciones
- `numpy/pandas`: Procesamiento de datos

---

## Uso

### 1. Configuración

Editar el archivo `config/config.yaml` para ajustar parámetros:

```yaml
# Rutas de datos
data:
  base_path: "audios/organized_by_age"
  output_path: "results"

# Parámetros de audio
audio:
  sample_rate: 22050
  duration: 3.0
  n_mels: 128

# Técnicas anti-overfitting
deep_learning:
  regularization:
    dropout_rate: 0.5
    l2_regularization: 0.001
    batch_normalization: true
```

### 2. Entrenar Modelos

#### Entrenamiento Completo

```bash
python train.py
```

#### Entrenamiento Rápido (para pruebas)

```bash
python train.py --max-samples 100
```

#### Entrenar Solo Modelos Convencionales

```bash
python train.py --skip-deep-learning
```

#### Entrenar Solo Deep Learning

```bash
python train.py --skip-conventional
```

### 3. Resultados

Los resultados se guardan automáticamente en `results/`:

- **Modelos**: `results/models/`
- **Gráficos**: `results/plots/`
- **Métricas**: `results/metrics/`

---

## Metodología

### Pipeline de Entrenamiento

1. **Carga de Datos**
   - Lectura de archivos de audio MP3
   - Normalización a 22050 Hz y 3 segundos
   - Mapeo de categorías originales a categorías objetivo

2. **Preprocesamiento**
   - Extracción de espectrogramas Mel
   - Extracción de características acústicas (MFCCs, ZCR, Spectral features)
   - Conversión a imágenes RGB para Deep Learning

3. **Data Augmentation** (Anti-Overfitting)
   - Time Stretching
   - Pitch Shifting
   - Adición de Ruido
   - Time Shifting
   - Balanceo de clases

4. **Modelos Convencionales**
   - Random Forest (200 estimadores)
   - SVM con kernel RBF
   - XGBoost (200 estimadores)

5. **Deep Learning con Transfer Learning**
   - Base: VGG16 pre-entrenado en ImageNet
   - Fine-tuning de últimas 4 capas
   - Capas personalizadas con regularización:
     - Dropout (0.5)
     - Batch Normalization
     - L2 Regularization (0.001)
   - Early Stopping (patience=15)
   - Learning Rate Schedule

6. **Evaluación**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC Curves
   - Precision-Recall Curves
   - Learning Curves

### Técnicas Anti-Overfitting Implementadas

✅ **Data Augmentation**: 4 técnicas de aumentación de audio
✅ **Dropout**: 0.5 en capas densas
✅ **Batch Normalization**: En todas las capas convolucionales
✅ **L2 Regularization**: 0.001 en kernels
✅ **Early Stopping**: Patience 15 epochs
✅ **Learning Rate Schedule**: Reducción adaptativa
✅ **Cross-Validation**: Para modelos convencionales
✅ **Class Balancing**: Mediante augmentation

---

## Resultados

### Métricas de Rendimiento

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Random Forest | TBD | TBD | TBD | TBD |
| SVM | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| VGG16 Transfer Learning | TBD | TBD | TBD | TBD |

*Nota: Ejecutar entrenamiento para obtener resultados*

### Visualizaciones Generadas

- Confusion Matrix por modelo
- ROC Curves multi-clase
- Precision-Recall Curves
- Learning Curves (Deep Learning)
- Feature Importance (Modelos convencionales)
- Class Distribution
- Sample Spectrograms
- Model Comparison

---

## Documentación

Documentación detallada disponible en `docs/`:

- **[INFORME_TECNICO.md](docs/INFORME_TECNICO.md)**: Informe técnico completo del proyecto
- **[METODOLOGIA.md](docs/METODOLOGIA.md)**: Metodología detallada y fundamentos teóricos
- **[RESULTADOS.md](docs/RESULTADOS.md)**: Análisis exhaustivo de resultados
- **[GUIA_USUARIO.md](docs/GUIA_USUARIO.md)**: Guía completa de uso

---

## Dataset

**Common Voice Corpus 23.0** (Portugués)
- Fuente: Mozilla Common Voice
- Idioma: Portugués (PT)
- Categorías originales: teens, twenties, thirties, fourties, fifties, sixties
- Total de muestras: ~23,000 audios

---

## Contacto y Soporte

Para preguntas o problemas:
- Revisar documentación en `docs/`
- Verificar configuración en `config/config.yaml`
- Revisar logs de entrenamiento

---

## Licencia

Este proyecto es parte del curso de Inteligencia Artificial II del ITM.

---

## Referencias

1. Mozilla Common Voice Dataset
2. Librosa: Audio Analysis Library
3. TensorFlow/Keras Documentation
4. scikit-learn Documentation
5. VGG16 Transfer Learning (ImageNet)

---

**Última Actualización**: Noviembre 2025
