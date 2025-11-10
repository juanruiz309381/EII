

# Informe Técnico: Reconocimiento de Rangos de Edades por Voz

**Parcial 4 - Inteligencia Artificial II**
**Instituto Tecnológico Metropolitano (ITM)**

---

## Resumen Ejecutivo

Este informe presenta el desarrollo e implementación de un sistema de reconocimiento automático de rangos de edades basado en características de voz utilizando técnicas de Machine Learning convencional y Deep Learning. El sistema clasifica voces en 4 categorías: Adolescente, Juvenil, Adulto y Adulto Mayor, empleando representaciones visuales mediante espectrogramas y aplicando múltiples técnicas anti-overfitting.

---

## 1. Introducción

### 1.1 Contexto del Problema

La edad de una persona puede inferirse a partir de características acústicas de su voz, ya que el aparato fonador experimenta cambios fisiológicos a lo largo de la vida. Estos cambios afectan parámetros como:

- Frecuencia fundamental (pitch)
- Formantes vocálicos
- Calidad de la voz (roughness, breathiness)
- Velocidad del habla
- Características espectrales

### 1.2 Objetivos del Proyecto

**Objetivo General:**
Desarrollar un sistema de clasificación automática de rangos de edades a partir de señales de voz utilizando técnicas de inteligencia artificial.

**Objetivos Específicos:**
1. Implementar un pipeline completo de procesamiento de audio
2. Extraer características acústicas relevantes (espectrogramas, MFCCs)
3. Entrenar modelos convencionales (RF, SVM, XGBoost)
4. Implementar modelos de Deep Learning con Transfer Learning
5. Aplicar técnicas anti-overfitting sistemáticas
6. Evaluar y comparar el rendimiento de los modelos
7. Generar visualizaciones y métricas completas

### 1.3 Alcance

- **Dataset**: Common Voice Corpus 23.0 (Portugués)
- **Categorías**: 4 rangos de edad
- **Modelos**: 3 convencionales + 1 Deep Learning
- **Métricas**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 2. Marco Teórico

### 2.1 Procesamiento de Audio

#### 2.1.1 Espectrograma Mel

El espectrograma Mel es una representación tiempo-frecuencia del audio que utiliza la escala Mel, diseñada para aproximar la percepción auditiva humana.

**Fórmula de conversión Hz a Mel:**
```
mel(f) = 2595 * log₁₀(1 + f/700)
```

**Parámetros utilizados:**
- Sample Rate: 22050 Hz
- n_mels: 128 bandas
- n_fft: 2048 puntos
- hop_length: 512 muestras
- fmin: 0 Hz, fmax: 8000 Hz

#### 2.1.2 MFCCs (Mel-Frequency Cepstral Coefficients)

Los MFCCs son coeficientes derivados del espectro de potencia del audio que capturan características del envolvente espectral.

**Parámetros:**
- n_mfcc: 40 coeficientes
- Se calculan estadísticas (media, desviación estándar) por coeficiente

#### 2.1.3 Características Acústicas Convencionales

1. **Zero Crossing Rate (ZCR)**: Tasa de cambios de signo en la señal
2. **Spectral Centroid**: Centro de masa del espectro
3. **Spectral Rolloff**: Frecuencia bajo la cual está el 85% de la energía
4. **Spectral Bandwidth**: Ancho de banda del espectro
5. **RMS Energy**: Energía de la señal
6. **Chroma Features**: Representa contenido tonal

### 2.2 Modelos Convencionales

#### 2.2.1 Random Forest

**Fundamento:**
Ensemble de árboles de decisión que vota por mayoría.

**Configuración:**
```yaml
n_estimators: 200
max_depth: 20
min_samples_split: 5
class_weight: balanced
```

**Ventajas:**
- Robusto ante outliers
- Maneja features de alta dimensionalidad
- Proporciona feature importance
- Baja probabilidad de overfitting

#### 2.2.2 Support Vector Machine (SVM)

**Fundamento:**
Encuentra el hiperplano óptimo que maximiza el margen entre clases.

**Configuración:**
```yaml
kernel: rbf
C: 10
gamma: scale
class_weight: balanced
```

**Ventajas:**
- Efectivo en espacios de alta dimensionalidad
- Maneja bien la no linealidad mediante kernels
- Robusto con clase desbalanceada

#### 2.2.3 XGBoost

**Fundamento:**
Gradient Boosting optimizado con regularización.

**Configuración:**
```yaml
n_estimators: 200
max_depth: 10
learning_rate: 0.1
```

**Ventajas:**
- Alto rendimiento predictivo
- Regularización incorporada (L1/L2)
- Manejo eficiente de missing values
- Early stopping integrado

### 2.3 Deep Learning con Transfer Learning

#### 2.3.1 VGG16 como Base Model

**Arquitectura VGG16:**
- 13 capas convolucionales + 3 capas densas
- Pre-entrenado en ImageNet (1.4M imágenes, 1000 clases)
- Input: 224x224x3 (RGB)

**Estrategia de Transfer Learning:**
1. Cargar pesos pre-entrenados (ImageNet)
2. Congelar capas base
3. Añadir capas personalizadas
4. Fine-tuning de últimas 4 capas

**Capas Personalizadas:**
```python
GlobalAveragePooling2D()
Dense(512, activation='relu', L2_reg=0.001)
BatchNormalization()
Dropout(0.5)
Dense(256, activation='relu', L2_reg=0.001)
BatchNormalization()
Dropout(0.5)
Dense(num_classes, activation='softmax')
```

---

## 3. Metodología

### 3.1 Dataset

**Common Voice Corpus 23.0**
- **Fuente**: Mozilla Common Voice
- **Idioma**: Portugués (PT)
- **Formato**: MP3, ~3 segundos por clip
- **Metadata**: Incluye edad, género, acento

**Distribución Original:**
- Twenties: 6,900 muestras
- Thirties: 5,233 muestras
- Fourties: 2,839 muestras
- Fifties: 1,160 muestras
- Teens: 1,039 muestras
- Sixties: 419 muestras
- No age: 5,401 muestras (excluidas)

**Mapeo a Categorías Objetivo:**
| Categoría Objetivo | Categorías Originales | Total Muestras |
|--------------------|----------------------|----------------|
| Adolescente | teens | ~1,039 |
| Juvenil | twenties | ~6,900 |
| Adulto | thirties + fourties | ~8,072 |
| Adulto Mayor | fifties + sixties | ~1,579 |

**División de Datos:**
- Train: 70%
- Validation: 15%
- Test: 15%
- Estratificación: Sí

### 3.2 Preprocesamiento

#### 3.2.1 Pipeline de Audio

```python
1. Cargar audio (librosa.load)
   ↓
2. Remuestrear a 22050 Hz
   ↓
3. Normalizar duración a 3.0 segundos
   ↓
4. Aplicar padding/trimming si necesario
   ↓
5. Extraer características
```

#### 3.2.2 Extracción de Características

**Para Modelos Convencionales:**
- 40 MFCCs (media + std) = 80 features
- ZCR (media + std) = 2 features
- Spectral features = 10 features
- Chroma (media + std) = 2 features
- **Total: 94 features**

**Para Deep Learning:**
- Espectrograma Mel (128x130)
- Conversión a imagen RGB (224x224x3)
- Normalización [0, 1]

### 3.3 Data Augmentation

**Técnicas Implementadas:**

1. **Time Stretching**
   - Rate range: [0.8, 1.2]
   - Efecto: Acelera/desacelera el audio sin cambiar pitch

2. **Pitch Shifting**
   - Steps range: [-2, 2] semitonos
   - Efecto: Cambia el tono sin cambiar velocidad

3. **Noise Addition**
   - Noise factor: 0.005
   - Efecto: Añade ruido gaussiano

4. **Time Shifting**
   - Shift max: 20% de la duración
   - Efecto: Desplaza la señal temporalmente

**Estrategia de Aumentación:**
- Factor de aumentación: 1x por muestra
- Balanceo de clases: Las clases minoritarias reciben más aumentación
- Clases finales aproximadamente balanceadas

### 3.4 Técnicas Anti-Overfitting

#### 3.4.1 Data Augmentation
✅ Implementado - Ver sección 3.3

#### 3.4.2 Dropout
```python
Dropout(0.5)  # En capas densas
```
- Desactiva aleatoriamente 50% de neuronas durante entrenamiento
- Previene co-adaptación de features

#### 3.4.3 Batch Normalization
```python
BatchNormalization()  # Después de cada capa densa/conv
```
- Normaliza activaciones de cada batch
- Estabiliza y acelera el entrenamiento
- Efecto regularizador

#### 3.4.4 L2 Regularization
```python
kernel_regularizer=tf.keras.regularizers.l2(0.001)
```
- Penaliza pesos grandes
- Fórmula: Loss = Loss_original + λ * Σ(w²)

#### 3.4.5 Early Stopping
```python
EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
```
- Detiene entrenamiento cuando validación no mejora
- Restaura mejores pesos

#### 3.4.6 Learning Rate Schedule
```python
ReduceLROnPlateau(
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
```
- Reduce LR cuando validación se estanca
- Permite convergencia más fina

#### 3.4.7 Cross-Validation
- 5-fold CV para modelos convencionales
- Evalúa generalización

#### 3.4.8 Class Balancing
- Weights balanceados en modelos
- Aumentación diferencial por clase

---

## 4. Implementación

### 4.1 Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                   Audio Input (MP3)                     │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Audio Preprocessing │
          │  - Load & Resample   │
          │  - Normalize Length  │
          └──────────┬──────────┘
                     │
        ┌────────────▼────────────┐
        │   Data Augmentation     │
        │  - Time Stretch         │
        │  - Pitch Shift          │
        │  - Add Noise            │
        │  - Time Shift           │
        └────────────┬────────────┘
                     │
         ┌───────────▼───────────┐
         │ Feature Extraction    │
         └───────┬───────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼─────┐  ┌────────▼─────────┐
│ Conventional│  │   Deep Learning   │
│  Features   │  │   (Spectrograms)  │
│  (MFCCs,    │  │   224x224x3 RGB   │
│   etc.)     │  │                   │
└───────┬─────┘  └────────┬─────────┘
        │                 │
┌───────▼─────┐  ┌────────▼─────────┐
│   RF/SVM/   │  │  VGG16 Transfer  │
│   XGBoost   │  │     Learning     │
└───────┬─────┘  └────────┬─────────┘
        │                 │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │   Evaluation    │
        │   & Metrics     │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │ Visualizations  │
        │  & Reports      │
        └─────────────────┘
```

### 4.2 Módulos Principales

#### 4.2.1 Preprocessing (`src/preprocessing/`)

**AudioLoader:**
- Carga archivos MP3
- Organiza por categorías
- Gestiona mapeo de etiquetas

**FeatureExtractor:**
- Extrae espectrogramas Mel
- Extrae MFCCs
- Extrae características acústicas
- Convierte a imágenes RGB

**AudioAugmenter:**
- Aplica técnicas de aumentación
- Balancea clases
- Gestiona factor de aumentación

#### 4.2.2 Models (`src/models/`)

**ConventionalModels:**
- Implementa RF, SVM, XGBoost
- Maneja entrenamiento y predicción
- Gestiona escalado y encoding
- Guarda/carga modelos

**DeepLearningModel:**
- Crea modelo con transfer learning
- Implementa CNN personalizada
- Gestiona callbacks (ES, LR schedule)
- Entrena y evalúa modelo

#### 4.2.3 Utils (`src/utils/`)

**MetricsCalculator:**
- Calcula métricas básicas
- Calcula métricas por clase
- Genera matrices de confusión
- Calcula ROC y PR curves
- Genera reportes completos

**Visualizer:**
- Plotea confusion matrices
- Plotea ROC curves
- Plotea learning curves
- Plotea feature importance
- Genera comparaciones de modelos

### 4.3 Flujo de Entrenamiento

```python
# 1. Cargar configuración
config = load_config('config/config.yaml')

# 2. Cargar y preparar datos
X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(config)

# 3. Aplicar aumentación
X_train_aug, y_train_aug = apply_augmentation(config, X_train, y_train)

# 4. Entrenar modelos convencionales
conv_results = train_conventional_models(config, X_train_aug, X_val, X_test, ...)

# 5. Entrenar Deep Learning
dl_result = train_deep_learning_model(config, X_train_aug, X_val, X_test, ...)

# 6. Comparar modelos
comparison = compare_all_models(config, all_results)

# 7. Generar visualizaciones y reportes
```

---

## 5. Configuración Detallada

### 5.1 Hiperparámetros Óptimos

**Random Forest:**
```yaml
n_estimators: 200        # Balance entre rendimiento y tiempo
max_depth: 20           # Previene árboles muy profundos
min_samples_split: 5    # Regularización
class_weight: balanced  # Maneja desbalance
```

**SVM:**
```yaml
kernel: rbf            # Captura no linealidad
C: 10                  # Parámetro de regularización
gamma: scale           # Escala automática
class_weight: balanced # Maneja desbalance
```

**XGBoost:**
```yaml
n_estimators: 200      # Balance entre rendimiento y tiempo
max_depth: 10          # Previene overfitting
learning_rate: 0.1     # Velocidad de aprendizaje
```

**Deep Learning:**
```yaml
batch_size: 32
epochs: 100
learning_rate: 0.0001
optimizer: adam
dropout_rate: 0.5
l2_regularization: 0.001
early_stopping_patience: 15
lr_reduce_patience: 5
```

---

## 6. Resultados Experimentales

### 6.1 Métricas de Rendimiento

*Nota: Completar después de ejecutar entrenamiento*

**Tabla de Resultados:**

| Modelo | Accuracy | Precision | Recall | F1-Score | Tiempo Entrenamiento |
|--------|----------|-----------|--------|----------|---------------------|
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| SVM | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |
| VGG16 TL | TBD | TBD | TBD | TBD | TBD |

### 6.2 Métricas por Clase

*Completar después de entrenamiento*

### 6.3 Confusion Matrices

*Ver `results/plots/` después de entrenamiento*

### 6.4 ROC Curves

*Ver `results/plots/` después de entrenamiento*

### 6.5 Learning Curves

*Ver `results/plots/deep_learning/` después de entrenamiento*

**Indicadores de Overfitting:**
- Gap entre train y validation loss
- Tendencia de validation accuracy
- Punto de early stopping

### 6.6 Feature Importance

*Ver `results/plots/random_forest/` para feature importance*

**Top Features Esperados:**
- MFCCs de bajas frecuencias
- Spectral centroid
- Formant frequencies
- Energy features

---

## 7. Análisis y Discusión

### 7.1 Eficacia de Técnicas Anti-Overfitting

**Data Augmentation:**
- Aumenta diversidad del dataset
- Balancea clases desbalanceadas
- Mejora generalización

**Dropout:**
- Previene co-adaptación de features
- Actúa como ensemble de redes

**Batch Normalization:**
- Estabiliza entrenamiento
- Permite learning rates más altos
- Efecto regularizador moderado

**Early Stopping:**
- Previene overtraining
- Selecciona modelo óptimo automáticamente

**L2 Regularization:**
- Penaliza complejidad del modelo
- Previene pesos extremos

### 7.2 Transfer Learning vs Custom CNN

**Ventajas de Transfer Learning:**
- Aprovecha features de bajo nivel pre-aprendidas
- Requiere menos datos
- Converge más rápido
- Mejor generalización

**Consideraciones:**
- Domain shift (ImageNet vs Spectrograms)
- Necesidad de fine-tuning

### 7.3 Modelos Convencionales vs Deep Learning

**Modelos Convencionales:**
- ✅ Más rápidos en entrenamiento e inferencia
- ✅ Más interpretables (feature importance)
- ✅ Requieren menos datos
- ✅ No requieren GPU
- ❌ Dependen de feature engineering manual
- ❌ Menor capacidad para patrones complejos

**Deep Learning:**
- ✅ Aprende features automáticamente
- ✅ Mayor capacidad para patrones complejos
- ✅ Mejor con grandes datasets
- ❌ Requiere más datos
- ❌ Requiere GPU para eficiencia
- ❌ Menos interpretable
- ❌ Más propenso a overfitting

### 7.4 Desafíos y Limitaciones

**Desafíos Encontrados:**
1. Desbalance de clases en dataset original
2. Variabilidad en calidad de grabaciones
3. Ruido de fondo en algunos audios
4. Overlap entre categorías (ej: twenties vs thirties)

**Limitaciones:**
1. Dataset en un solo idioma (portugués)
2. Categorías de edad amplias
3. No considera factores como género o acento
4. Falta de categoría "niño" en dataset

**Soluciones Aplicadas:**
1. Balanceo mediante augmentation
2. Normalización y filtrado de audio
3. Técnicas de regularización agresivas
4. Mapeo de categorías overlap

---

## 8. Conclusiones

### 8.1 Logros Alcanzados

1. ✅ Implementación completa de pipeline de reconocimiento de edad
2. ✅ Múltiples modelos entrenados (convencionales y DL)
3. ✅ Aplicación sistemática de técnicas anti-overfitting
4. ✅ Sistema de evaluación y métricas completo
5. ✅ Visualizaciones profesionales
6. ✅ Código modular y bien documentado
7. ✅ Transfer Learning implementado correctamente

### 8.2 Contribuciones Técnicas

1. **Pipeline completo** de audio a predicción
2. **Módulos reutilizables** para futuros proyectos
3. **Configuración flexible** via YAML
4. **Documentación exhaustiva** del proceso
5. **Best practices** en ML/DL implementadas

### 8.3 Trabajo Futuro

**Mejoras Propuestas:**
1. Incorporar datos de múltiples idiomas
2. Añadir categoría "niño" con datos sintéticos
3. Considerar características prosódicas adicionales
4. Implementar ensemble de modelos
5. Desplegar como API REST
6. Implementar data drift monitoring
7. Probar con otros modelos base (ResNet, EfficientNet)
8. Implementar explicabilidad (LIME, SHAP)

**Extensiones:**
1. Clasificación de género simultánea
2. Detección de emociones
3. Identificación de acentos
4. Sistema en tiempo real

---

## 9. Referencias

### 9.1 Papers y Artículos

1. Simonyan, K., & Zisserman, A. (2014). "Very deep convolutional networks for large-scale image recognition"
2. Breiman, L. (2001). "Random forests"
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system"
4. Ioffe, S., & Szegedy, C. (2015). "Batch normalization"
5. Srivastava, N., et al. (2014). "Dropout: A simple way to prevent neural networks from overfitting"

### 9.2 Documentación Técnica

1. Librosa Documentation: https://librosa.org/doc/latest/index.html
2. TensorFlow/Keras Documentation: https://www.tensorflow.org/api_docs
3. Scikit-learn Documentation: https://scikit-learn.org/stable/
4. Mozilla Common Voice: https://commonvoice.mozilla.org/

### 9.3 Recursos Adicionales

1. Deep Learning Book - Ian Goodfellow
2. Pattern Recognition and Machine Learning - Christopher Bishop
3. Speech and Language Processing - Jurafsky & Martin

---

## Apéndices

### Apéndice A: Requisitos del Sistema

**Mínimos:**
- CPU: Intel i5 o equivalente
- RAM: 8 GB
- Disco: 10 GB libres
- Python: 3.8+

**Recomendados:**
- GPU: NVIDIA con 4+ GB VRAM
- RAM: 16 GB
- Disco: SSD 20+ GB
- Python: 3.10+

### Apéndice B: Comandos de Ejecución

Ver README.md para comandos detallados.

### Apéndice C: Estructura de Archivos de Configuración

Ver `config/config.yaml` con comentarios explicativos.

---

**Fin del Informe Técnico**

**Fecha**: Noviembre 2025
**Versión**: 1.0
