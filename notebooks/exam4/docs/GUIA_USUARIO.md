# Guía de Usuario - Reconocimiento de Edades por Voz

## Índice

1. [Introducción](#introducción)
2. [Instalación](#instalación)
3. [Configuración](#configuración)
4. [Uso Básico](#uso-básico)
5. [Uso Avanzado](#uso-avanzado)
6. [Interpretación de Resultados](#interpretación-de-resultados)
7. [Troubleshooting](#troubleshooting)

---

## Introducción

Esta guía proporciona instrucciones paso a paso para usar el sistema de reconocimiento de edades por voz.

### ¿Qué hace el sistema?

El sistema analiza archivos de audio de voz y predice el rango de edad del hablante en una de 4 categorías:
- **Adolescente** (13-19 años)
- **Juvenil** (20-29 años)
- **Adulto** (30-49 años)
- **Adulto Mayor** (50+ años)

---

## Instalación

### Paso 1: Requisitos Previos

Asegúrate de tener instalado:
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- (Opcional) GPU NVIDIA con CUDA para Deep Learning

### Paso 2: Instalar Dependencias

```bash
# Navegar al directorio del proyecto
cd /ruta/al/proyecto/exam4

# Crear entorno virtual (recomendado)
python -m venv venv

# Activar entorno virtual
# En Linux/Mac:
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 3: Verificar Instalación

```bash
# Verificar que Python puede importar los módulos
python -c "import librosa; import tensorflow; print('OK')"
```

Si ves "OK", la instalación fue exitosa.

---

## Configuración

### Archivo de Configuración

El archivo `config/config.yaml` controla todos los parámetros del sistema.

#### Configuración Básica

```yaml
# Rutas de datos
data:
  base_path: "audios/organized_by_age"  # Carpeta con audios
  output_path: "results"                 # Carpeta de resultados
```

#### Parámetros de Audio

```yaml
audio:
  sample_rate: 22050  # Frecuencia de muestreo
  duration: 3.0       # Duración en segundos
  n_mels: 128         # Número de bandas Mel
```

#### Activar/Desactivar Augmentation

```yaml
augmentation:
  enabled: true  # Cambiar a false para desactivar
```

#### Configurar Técnicas Anti-Overfitting

```yaml
deep_learning:
  regularization:
    dropout_rate: 0.5              # Ajustar entre 0.3-0.7
    l2_regularization: 0.001       # Ajustar entre 0.0001-0.01
    batch_normalization: true      # true/false
```

---

## Uso Básico

### Entrenamiento Completo

Entrenar todos los modelos con configuración por defecto:

```bash
python train.py
```

Esto ejecutará:
1. Carga de datos
2. Preprocesamiento
3. Data augmentation
4. Entrenamiento de modelos convencionales (RF, SVM, XGBoost)
5. Entrenamiento de Deep Learning (VGG16 Transfer Learning)
6. Evaluación y generación de métricas
7. Generación de visualizaciones

**Tiempo estimado:** 30-60 minutos (dependiendo de hardware)

### Visualizar Resultados

Los resultados se guardan automáticamente en `results/`:

```bash
# Ver estructura de resultados
ls -R results/

# Abrir gráficos
# En Linux:
xdg-open results/plots/model_comparison.png

# En Mac:
open results/plots/model_comparison.png

# En Windows:
start results/plots/model_comparison.png
```

---

## Uso Avanzado

### Entrenamiento Rápido (Para Pruebas)

Entrenar con un subset pequeño de datos:

```bash
python train.py --max-samples 100
```

Limita el entrenamiento a 100 muestras por categoría. Útil para:
- Probar el código
- Verificar que todo funciona
- Desarrollo rápido

**Tiempo estimado:** 5-10 minutos

### Entrenar Solo Modelos Convencionales

```bash
python train.py --skip-deep-learning
```

Útil cuando:
- No tienes GPU
- Quieres resultados rápidos
- Solo te interesan modelos interpretables

**Tiempo estimado:** 10-15 minutos

### Entrenar Solo Deep Learning

```bash
python train.py --skip-conventional
```

Útil cuando:
- Ya entrenaste modelos convencionales
- Solo quieres experimentar con Deep Learning
- Tienes GPU y quieres aprovecharla

**Tiempo estimado:** 20-40 minutos

### Usar Configuración Personalizada

```bash
# Crear copia de configuración
cp config/config.yaml config/mi_config.yaml

# Editar mi_config.yaml con tus parámetros

# Usar configuración personalizada
python train.py --config config/mi_config.yaml
```

### Monitorear Entrenamiento en Tiempo Real

Para Deep Learning, puedes usar TensorBoard:

```bash
# En terminal separada, ejecutar:
tensorboard --logdir results/models/deep_learning/tensorboard_logs

# Abrir navegador en:
# http://localhost:6006
```

---

## Interpretación de Resultados

### Métricas de Clasificación

**Accuracy (Exactitud):**
- Porcentaje de predicciones correctas
- Métrica general de rendimiento
- Valor ideal: Cercano a 1.0 (100%)

**Precision (Precisión):**
- De las predicciones positivas, cuántas fueron correctas
- Importante cuando falsos positivos son costosos
- Fórmula: TP / (TP + FP)

**Recall (Sensibilidad):**
- De los casos reales, cuántos detectamos
- Importante cuando falsos negativos son costosos
- Fórmula: TP / (TP + FN)

**F1-Score:**
- Media armónica de Precision y Recall
- Balancea ambas métricas
- Fórmula: 2 * (Precision * Recall) / (Precision + Recall)

### Confusion Matrix

Ejemplo de interpretación:

```
              Predicted
           Adol  Juv  Adult  Mayor
Actual Adol [ 85    5    3     0 ]
       Juv  [  4   78    6     2 ]
       Adult[  2    8   82     3 ]
       Mayor[  0    1    5    89 ]
```

- **Diagonal**: Predicciones correctas (valores altos = bueno)
- **Fuera de diagonal**: Errores (valores bajos = bueno)
- **Confusiones comunes**: Valores altos cerca de la diagonal

### Learning Curves

**Train vs Validation Loss:**

```
   Loss
    │
    │  ╱╲ Train
    │ ╱  ╲___________
    │╱    ╲ Validation
    └─────────────────── Epochs
```

**Interpretación:**
- **Gap pequeño**: Modelo generaliza bien ✅
- **Gap grande**: Overfitting ⚠️
- **Ambas altas**: Underfitting ⚠️
- **Validation aumenta**: Overfitting severo ❌

### ROC Curves

```
TPR │
    │     ╱
    │    ╱
    │   ╱
    │  ╱
    │ ╱__________ Random (AUC=0.5)
    └──────────── FPR
```

**AUC (Area Under Curve):**
- 1.0: Clasificador perfecto ✅
- 0.9-1.0: Excelente ✅
- 0.8-0.9: Muy bueno ✅
- 0.7-0.8: Bueno ✅
- 0.6-0.7: Regular ⚠️
- 0.5-0.6: Malo ❌
- 0.5: Aleatorio ❌

### Feature Importance

Para modelos convencionales (Random Forest, XGBoost):

```
mfcc_0_mean     ████████████ 0.15
spectral_cent   ███████████  0.12
mfcc_1_mean     ██████████   0.10
rms_mean        ████████     0.08
...
```

**Interpretación:**
- Features con mayor importancia contribuyen más a la predicción
- Útil para entender qué características del audio son más relevantes
- Puede guiar feature engineering

---

## Troubleshooting

### Problema: Error al cargar archivos de audio

**Error:**
```
FileNotFoundError: audios/organized_by_age/...
```

**Solución:**
1. Verificar que la carpeta de audios existe
2. Verificar ruta en `config/config.yaml`
3. Ejecutar script de organización: `python preprocessing/organize_audios_by_age.py`

### Problema: Out of Memory (OOM)

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Soluciones:**
1. Reducir `batch_size` en config:
   ```yaml
   training:
     batch_size: 16  # Reducir de 32
   ```

2. Usar menos muestras:
   ```bash
   python train.py --max-samples 500
   ```

3. Cerrar otras aplicaciones

4. Entrenar solo modelos convencionales:
   ```bash
   python train.py --skip-deep-learning
   ```

### Problema: Entrenamiento muy lento

**Soluciones:**
1. **Sin GPU disponible:**
   - Usar modelos convencionales: `--skip-deep-learning`
   - Reducir datos: `--max-samples 500`

2. **Con GPU pero lenta:**
   - Verificar que TensorFlow usa GPU:
     ```python
     python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
     ```
   - Instalar drivers CUDA correctos

3. **Reducir complejidad:**
   - Reducir epochs: `epochs: 50` en config
   - Reducir tamaño de imagen: `image_size: [128, 128]`

### Problema: Accuracy muy baja

**Posibles causas y soluciones:**

1. **Datos insuficientes:**
   - Activar augmentation: `enabled: true`
   - Aumentar factor de augmentation

2. **Overfitting:**
   - Aumentar dropout: `dropout_rate: 0.6`
   - Activar batch normalization
   - Aumentar L2 regularization

3. **Underfitting:**
   - Entrenar más epochs
   - Aumentar complejidad del modelo
   - Verificar learning rate

4. **Desbalance de clases:**
   - Verificar distribución en plots
   - Activar class balancing en augmentation

### Problema: Módulo no encontrado

**Error:**
```
ModuleNotFoundError: No module named 'librosa'
```

**Solución:**
```bash
# Verificar que estás en el entorno virtual
which python

# Reinstalar dependencias
pip install -r requirements.txt

# Si persiste, instalar manualmente
pip install librosa tensorflow scikit-learn
```

### Problema: CUDA / GPU no funciona

**Verificar instalación:**
```bash
# Verificar CUDA
nvcc --version

# Verificar TensorFlow ve GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Si no detecta GPU:**
1. Instalar TensorFlow para GPU:
   ```bash
   pip install tensorflow[and-cuda]
   ```

2. Verificar drivers NVIDIA actualizados

3. Entrenar sin GPU (más lento):
   ```bash
   python train.py --skip-deep-learning
   ```

---

## FAQ

**P: ¿Cuánto tiempo tarda el entrenamiento completo?**

R: Depende del hardware:
- Con GPU: 30-60 minutos
- Sin GPU (solo conventional): 10-20 minutos
- Sin GPU (con DL): 2-4 horas

**P: ¿Puedo usar mis propios audios?**

R: Sí, organízalos en la estructura requerida:
```
audios/organized_by_age/
├── adolescente/
├── juvenil/
├── adulto/
└── adulto_mayor/
```

**P: ¿Qué formato de audio soporta?**

R: MP3, WAV, FLAC. Cualquier formato que soporte librosa.

**P: ¿Necesito GPU?**

R: No es obligatorio, pero:
- Modelos convencionales: No necesitan GPU
- Deep Learning: Recomendado GPU (30x más rápido)

**P: ¿Puedo usar solo un modelo específico?**

R: Sí, modifica el código o usa flags:
```bash
python train.py --skip-deep-learning  # Solo conventional
python train.py --skip-conventional   # Solo DL
```

**P: ¿Cómo interpreto si hay overfitting?**

R: Mira las learning curves:
- Gap grande entre train/val loss = overfitting
- Validation loss aumenta = overfitting
- Early stopping activado temprano = posible overfitting

**P: ¿Puedo cambiar las categorías de edad?**

R: Sí, edita `config.yaml`:
```yaml
category_mapping:
  "teens": "tu_categoria"
  ...
```

---

## Contacto y Soporte

Para problemas no cubiertos en esta guía:

1. Revisar logs en consola
2. Revisar `INFORME_TECNICO.md` para detalles técnicos
3. Verificar configuración en `config/config.yaml`

---

**Última Actualización**: Noviembre 2025
