# Configuración del Entorno Virtual Python

Guía para crear y configurar un entorno virtual Python para el proyecto.

## Opción 1: venv (Recomendado - viene con Python)

### Linux / macOS

```bash
# 1. Navegar al directorio del proyecto
cd parcial5_llm

# 2. Crear entorno virtual
python3 -m venv venv

# 3. Activar entorno virtual
source venv/bin/activate

# 4. Actualizar pip
pip install --upgrade pip

# 5. Instalar dependencias
pip install -r requirements.txt

# 6. Verificar instalación
pip list
```

### Windows

```cmd
# 1. Navegar al directorio del proyecto
cd parcial5_llm

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
venv\Scripts\activate

# 4. Actualizar pip
pip install --upgrade pip

# 5. Instalar dependencias
pip install -r requirements.txt

# 6. Verificar instalación
pip list
```

## Opción 2: conda (Si tienes Anaconda/Miniconda)

```bash
# 1. Crear entorno con Python 3.10
conda create -n llm_eval python=3.10

# 2. Activar entorno
conda activate llm_eval

# 3. Instalar dependencias desde requirements.txt
pip install -r requirements.txt

# Alternativa: instalar paquetes principales con conda
conda install jupyter pandas numpy matplotlib seaborn
pip install -r requirements.txt
```

## Verificar Instalación

```bash
# Con el entorno activado
python3 verify_setup.py
```

## Uso del Entorno

### Activar entorno cada vez que trabajes

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```cmd
venv\Scripts\activate
```

### Desactivar entorno cuando termines

```bash
deactivate
```

### Usar Jupyter con el entorno

```bash
# Con entorno activado
jupyter notebook
# o
jupyter lab
```

## Añadir Nuevas Dependencias

```bash
# Instalar nuevo paquete
pip install nombre_paquete

# Actualizar requirements.txt
pip freeze > requirements.txt
```

## Troubleshooting

### Error: "No module named 'venv'"

**Solución:** Instalar python3-venv
```bash
# Ubuntu/Debian
sudo apt install python3-venv

# Fedora/RHEL
sudo dnf install python3-virtualenv
```

### Error: "pip: command not found"

**Solución:** Instalar pip
```bash
# Ubuntu/Debian
sudo apt install python3-pip

# macOS (con Homebrew)
brew install python3
```

### Jupyter no encuentra el kernel

**Solución:** Registrar el entorno como kernel
```bash
# Con entorno activado
python3 -m ipykernel install --user --name=llm_eval --display-name="Python (LLM Eval)"
```

### Error de permisos al instalar

**Solución:** Usar flag --user (solo si no estás en venv)
```bash
pip install --user -r requirements.txt
```

## Estructura Recomendada

```
parcial5_llm/
├── venv/                    # Entorno virtual (no subir a git)
├── requirements.txt         # Dependencias
├── notebooks/              # Notebooks Jupyter
├── scripts/                # Scripts Python
├── results/                # Resultados generados
└── .gitignore              # Ignorar venv/
```

## .gitignore Recomendado

Crea un archivo `.gitignore` con:

```
# Entorno virtual
venv/
env/
ENV/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Resultados grandes
results/*.json
results/*.csv

# Sistema
.DS_Store
Thumbs.db
```

## Exportar Entorno para Compartir

```bash
# Crear requirements.txt limpio
pip freeze > requirements.txt

# Alternativa: solo dependencias directas
pip list --format=freeze > requirements-minimal.txt
```

## Recursos Adicionales

- **Documentación venv:** https://docs.python.org/3/library/venv.html
- **Guía pip:** https://pip.pypa.io/en/stable/user_guide/
- **Conda cheatsheet:** https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html

---

**¡Listo! Ahora tienes un entorno Python aislado para el proyecto. 🐍**
