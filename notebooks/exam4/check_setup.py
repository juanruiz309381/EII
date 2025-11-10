#!/usr/bin/env python3
"""
Script de verificación de instalación y configuración.

Verifica que todas las dependencias estén instaladas y que
el entorno esté correctamente configurado.
"""

import sys
import os

def check_python_version():
    """Verifica versión de Python."""
    print("="*60)
    print("Verificando Python...")
    print("="*60)

    version = sys.version_info
    print(f"Versión de Python: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 8:
        print("✅ Versión de Python OK (3.8+)")
        return True
    else:
        print("❌ Se requiere Python 3.8 o superior")
        return False


def check_dependencies():
    """Verifica que todas las dependencias estén instaladas."""
    print("\n" + "="*60)
    print("Verificando Dependencias...")
    print("="*60)

    dependencies = {
        'librosa': 'Audio processing',
        'soundfile': 'Audio I/O',
        'numpy': 'Computación numérica',
        'pandas': 'Manipulación de datos',
        'sklearn': 'Machine Learning',
        'xgboost': 'Gradient Boosting',
        'tensorflow': 'Deep Learning',
        'keras': 'Deep Learning API',
        'matplotlib': 'Visualización',
        'seaborn': 'Visualización estadística',
        'yaml': 'Configuración',
        'tqdm': 'Progress bars'
    }

    all_ok = True

    for package, description in dependencies.items():
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)

            print(f"✅ {package:15} - {description}")
        except ImportError:
            print(f"❌ {package:15} - {description} (NO INSTALADO)")
            all_ok = False

    return all_ok


def check_gpu():
    """Verifica disponibilidad de GPU."""
    print("\n" + "="*60)
    print("Verificando GPU...")
    print("="*60)

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            print(f"✅ GPU detectada: {len(gpus)} dispositivo(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            return True
        else:
            print("⚠️  No se detectó GPU")
            print("   El entrenamiento será más lento (CPU only)")
            print("   Deep Learning puede tardar varias horas")
            return False
    except Exception as e:
        print(f"❌ Error verificando GPU: {e}")
        return False


def check_directories():
    """Verifica estructura de directorios."""
    print("\n" + "="*60)
    print("Verificando Estructura de Directorios...")
    print("="*60)

    required_dirs = [
        'config',
        'src',
        'src/preprocessing',
        'src/models',
        'src/utils',
        'docs',
        'audios',
        'results'
    ]

    all_ok = True

    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory} (NO EXISTE)")
            all_ok = False

    return all_ok


def check_config():
    """Verifica archivo de configuración."""
    print("\n" + "="*60)
    print("Verificando Configuración...")
    print("="*60)

    config_path = 'config/config.yaml'

    if not os.path.exists(config_path):
        print(f"❌ {config_path} no encontrado")
        return False

    try:
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"✅ Configuración cargada correctamente")

        # Verificar campos importantes
        required_fields = ['data', 'audio', 'features', 'deep_learning']

        for field in required_fields:
            if field in config:
                print(f"   ✅ {field}")
            else:
                print(f"   ❌ {field} (FALTA)")
                return False

        return True

    except Exception as e:
        print(f"❌ Error al cargar configuración: {e}")
        return False


def check_dataset():
    """Verifica dataset."""
    print("\n" + "="*60)
    print("Verificando Dataset...")
    print("="*60)

    dataset_path = 'audios/organized_by_age'

    if not os.path.exists(dataset_path):
        print(f"⚠️  {dataset_path} no encontrado")
        print("   Ejecute el script de organización:")
        print("   python preprocessing/organize_audios_by_age.py")
        return False

    categories = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties']
    found_categories = []

    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if os.path.exists(category_path):
            mp3_files = [f for f in os.listdir(category_path) if f.endswith('.mp3')]
            found_categories.append(category)
            print(f"✅ {category:10} - {len(mp3_files)} archivos")

    if len(found_categories) >= 4:
        print(f"\n✅ Dataset OK ({len(found_categories)} categorías encontradas)")
        return True
    else:
        print(f"\n⚠️  Dataset incompleto ({len(found_categories)} categorías)")
        return False


def print_summary(results):
    """Imprime resumen de verificación."""
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)

    checks = [
        ("Python Version", results['python']),
        ("Dependencias", results['dependencies']),
        ("GPU", results['gpu']),
        ("Directorios", results['directories']),
        ("Configuración", results['config']),
        ("Dataset", results['dataset'])
    ]

    all_critical_ok = True

    for name, status in checks:
        if status:
            print(f"✅ {name}")
        else:
            if name in ['Python Version', 'Dependencias', 'Directorios', 'Configuración']:
                print(f"❌ {name} (CRÍTICO)")
                all_critical_ok = False
            else:
                print(f"⚠️  {name} (OPCIONAL)")

    print("\n" + "="*60)

    if all_critical_ok:
        print("✅ SISTEMA LISTO PARA ENTRENAR")
        print("\nPara comenzar, ejecute:")
        print("  python train.py")
        print("\nPara entrenamiento rápido (pruebas):")
        print("  python train.py --max-samples 100")

        if not results['gpu']:
            print("\n⚠️  NOTA: Sin GPU, considere:")
            print("  python train.py --skip-deep-learning")

        if not results['dataset']:
            print("\n⚠️  NOTA: Dataset no encontrado, organice primero:")
            print("  python preprocessing/organize_audios_by_age.py")

        return 0
    else:
        print("❌ SISTEMA NO LISTO")
        print("\nCorrija los errores críticos antes de continuar.")
        print("Consulte README.md y GUIA_USUARIO.md para más información.")
        return 1


def main():
    """Función principal."""
    print("\n" + "#"*60)
    print("# Verificación de Sistema - Age Recognition")
    print("#"*60 + "\n")

    results = {
        'python': check_python_version(),
        'dependencies': check_dependencies(),
        'gpu': check_gpu(),
        'directories': check_directories(),
        'config': check_config(),
        'dataset': check_dataset()
    }

    exit_code = print_summary(results)

    print("\n" + "#"*60)
    print("# Verificación Completa")
    print("#"*60 + "\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
