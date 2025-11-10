# 🚀 Guía de Inicio Rápido

## Reconocimiento de Edades por Voz - Parcial 4

---

## ⚡ En 5 Minutos

### 1. Verificar Instalación

```bash
python check_setup.py
```

Este script verifica:
- ✅ Python 3.8+
- ✅ Dependencias instaladas
- ✅ Estructura de directorios
- ✅ Configuración
- ✅ Dataset disponible
- 🔍 GPU (opcional)

### 2. Instalar Dependencias (si faltan)

```bash
pip install -r requirements.txt
```

### 3. Entrenar Modelos

**Entrenamiento Completo (~30-60 min con GPU):**
```bash
python train.py
```

**Entrenamiento Rápido para Pruebas (~5-10 min):**
```bash
python train.py --max-samples 100
```

**Solo Modelos Convencionales (~10-15 min):**
```bash
python train.py --skip-deep-learning
```

### 4. Ver Resultados

```bash
# Listar resultados
ls -R results/

# Ver gráficos (Linux)
xdg-open results/plots/model_comparison.png

# Ver logs
tail -f results/training.log
```

---

## 📊 Monitoreo en Tiempo Real

### TensorBoard (Deep Learning)

```bash
# Terminal 1: Iniciar TensorBoard
tensorboard --logdir results/models/deep_learning/tensorboard_logs

# Terminal 2: Iniciar entrenamiento
python train.py
```

Abrir navegador en: http://localhost:6006

---

## 📁 Estructura de Resultados

```
results/
├── models/
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── xgboost.pkl
│   └── deep_learning/
│       ├── best_model.h5
│       └── final_model.h5
├── plots/
│   ├── model_comparison.png
│   ├── class_distribution.png
│   ├── random_forest/
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   └── deep_learning/
│       ├── confusion_matrix.png
│       └── learning_curves.png
└── metrics/
    ├── random_forest_report.json
    ├── svm_report.json
    ├── xgboost_report.json
    └── deep_learning_report.json
```

---

## 🎯 Comandos Útiles

### Verificación

```bash
# Verificar instalación completa
python check_setup.py

# Verificar versión de Python
python --version

# Verificar GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Entrenamiento

```bash
# Entrenar todos los modelos
python train.py

# Entrenar con config personalizada
python train.py --config config/mi_config.yaml

# Entrenar con subset de datos
python train.py --max-samples 500

# Solo conventional
python train.py --skip-deep-learning

# Solo deep learning
python train.py --skip-conventional
```

### Análisis de Resultados

```bash
# Ver estructura de resultados
tree results/

# Contar archivos generados
find results/ -type f | wc -l

# Ver métricas JSON
cat results/metrics/random_forest_report.json | python -m json.tool

# Buscar mejores resultados
grep -r "accuracy" results/metrics/
```

---

## 🐛 Troubleshooting Rápido

### Error: ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### Error: Out of Memory

```bash
# Reducir batch size en config/config.yaml
# O usar menos datos:
python train.py --max-samples 500
```

### Entrenamiento Muy Lento

```bash
# Sin GPU, usar solo conventional:
python train.py --skip-deep-learning
```

### Dataset No Encontrado

```bash
# Organizar dataset primero:
python preprocessing/organize_audios_by_age.py
```

---

## 📖 Documentación Completa

- **README.md**: Visión general y documentación principal
- **docs/INFORME_TECNICO.md**: Informe técnico completo (40+ páginas)
- **docs/INFORME_PROYECTO.md**: Resumen del proyecto y logros
- **docs/GUIA_USUARIO.md**: Guía detallada de usuario
- **config/config.yaml**: Configuración con comentarios

---

## 🎓 Resumen del Proyecto

### Objetivo

Clasificar voces en 4 categorías de edad usando ML y DL:
- Adolescente (13-19)
- Juvenil (20-29)
- Adulto (30-49)
- Adulto Mayor (50+)

### Modelos Implementados

1. **Random Forest** (200 estimadores)
2. **SVM** (kernel RBF)
3. **XGBoost** (200 estimadores)
4. **VGG16 Transfer Learning** (ImageNet)

### Técnicas Anti-Overfitting

✅ Data Augmentation (4 técnicas)
✅ Dropout (0.5)
✅ Batch Normalization
✅ L2 Regularization (0.001)
✅ Early Stopping (patience 15)
✅ Learning Rate Schedule
✅ Cross-Validation (5-fold)
✅ Class Balancing

### Métricas Evaluadas

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curves y AUC
- Precision-Recall Curves
- Learning Curves (DL)
- Feature Importance (Conventional)

---

## 🚀 Siguiente Paso

```bash
# ¡Comienza ahora!
python train.py --max-samples 100
```

**Tiempo estimado**: 5-10 minutos

---

## 💡 Tips

- Usa `--max-samples` para pruebas rápidas
- Monitorea con TensorBoard para DL
- Revisa logs en tiempo real: `tail -f results/training.log`
- GPU acelera 30x el entrenamiento de DL
- Sin GPU, usa `--skip-deep-learning`

---

**¿Dudas?** Consulta `docs/GUIA_USUARIO.md`

**¿Detalles técnicos?** Lee `docs/INFORME_TECNICO.md`

---

**¡Buena suerte con el entrenamiento! 🎉**
