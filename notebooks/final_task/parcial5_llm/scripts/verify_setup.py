#!/usr/bin/env python3
"""
Script de Verificación del Sistema
Verifica que todos los componentes estén instalados y funcionando correctamente
"""

import subprocess
import sys
import os
from pathlib import Path

class Colors:
    """Códigos ANSI para colores en terminal"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Imprime un encabezado formateado"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text:^60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}\n")

def print_success(text):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")

def print_error(text):
    """Imprime mensaje de error"""
    print(f"{Colors.RED}✗{Colors.RESET} {text}")

def print_warning(text):
    """Imprime mensaje de advertencia"""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")

def print_info(text):
    """Imprime mensaje informativo"""
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {text}")

def check_command(command, name):
    """Verifica si un comando existe"""
    try:
        result = subprocess.run(
            [command, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print_success(f"{name} instalado: {version}")
            return True
        else:
            print_error(f"{name} no encontrado")
            return False
    except FileNotFoundError:
        print_error(f"{name} no instalado")
        return False
    except Exception as e:
        print_error(f"Error verificando {name}: {str(e)}")
        return False

def check_python_packages():
    """Verifica paquetes Python necesarios"""
    required_packages = [
        'jupyter',
        'pandas',
        'matplotlib',
        'seaborn'
    ]

    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"Paquete Python '{package}' instalado")
        except ImportError:
            print_error(f"Paquete Python '{package}' NO instalado")
            all_installed = False

    return all_installed

def check_ollama_models():
    """Verifica modelos de Ollama instalados"""
    required_models = [
        'llama3.2:1b',
        'deepseek-r1:1.5b',
        'qwen2.5:0.5b'
    ]

    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            output = result.stdout
            all_models_found = True

            for model in required_models:
                if model in output:
                    print_success(f"Modelo '{model}' instalado")
                else:
                    print_error(f"Modelo '{model}' NO instalado")
                    all_models_found = False

            return all_models_found
        else:
            print_error("No se pudo listar modelos de Ollama")
            return False

    except Exception as e:
        print_error(f"Error verificando modelos: {str(e)}")
        return False

def check_file_structure():
    """Verifica estructura de directorios"""
    base_dir = Path(__file__).parent.parent

    required_dirs = [
        'notebooks',
        'scripts',
        'results',
        'docs',
        'assets'
    ]

    required_files = [
        'notebooks/LLM_Evaluation.ipynb',
        'scripts/test_llm_models.py',
        'docs/informe_tecnico.md',
        'docs/script_video.md',
        'QUICK_START.md'
    ]

    print_info("Verificando estructura de archivos...")

    all_exist = True

    # Verificar directorios
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print_success(f"Directorio '{dir_name}/' existe")
        else:
            print_error(f"Directorio '{dir_name}/' NO existe")
            all_exist = False

    # Verificar archivos
    for file_name in required_files:
        file_path = base_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print_success(f"Archivo '{file_name}' existe ({size} bytes)")
        else:
            print_error(f"Archivo '{file_name}' NO existe")
            all_exist = False

    return all_exist

def test_ollama_query():
    """Prueba una consulta simple a Ollama"""
    print_info("Probando consulta a Ollama con llama3.2:1b...")

    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3.2:1b', 'Di solo "Hola"'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0 and result.stdout.strip():
            print_success(f"Consulta exitosa. Respuesta: {result.stdout.strip()[:50]}...")
            return True
        else:
            print_error("Consulta falló o sin respuesta")
            return False

    except subprocess.TimeoutExpired:
        print_error("Consulta excedió tiempo límite (30s)")
        return False
    except Exception as e:
        print_error(f"Error en consulta: {str(e)}")
        return False

def check_system_resources():
    """Verifica recursos del sistema"""
    print_info("Verificando recursos del sistema...")

    try:
        # RAM disponible
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemAvailable' in line:
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / (1024 ** 2)
                    if mem_gb >= 4:
                        print_success(f"RAM disponible: {mem_gb:.1f} GB")
                    else:
                        print_warning(f"RAM disponible: {mem_gb:.1f} GB (recomendado >= 4 GB)")
                    break

        # Espacio en disco
        result = subprocess.run(
            ['df', '-h', '.'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            if len(lines) >= 2:
                parts = lines[1].split()
                available = parts[3]
                print_success(f"Espacio disponible: {available}")

        return True

    except Exception as e:
        print_warning(f"No se pudo verificar recursos: {str(e)}")
        return True  # No crítico

def main():
    """Función principal"""
    print_header("VERIFICACIÓN DEL SISTEMA - PARCIAL 5 LLM")

    checks = []

    # 1. Verificar comandos básicos
    print_header("1. Verificando Comandos Básicos")
    checks.append(("Python", check_command('python3', 'Python')))
    checks.append(("Ollama", check_command('ollama', 'Ollama')))

    # 2. Verificar paquetes Python
    print_header("2. Verificando Paquetes Python")
    checks.append(("Paquetes Python", check_python_packages()))

    # 3. Verificar modelos Ollama
    print_header("3. Verificando Modelos Ollama")
    checks.append(("Modelos Ollama", check_ollama_models()))

    # 4. Verificar estructura de archivos
    print_header("4. Verificando Estructura de Archivos")
    checks.append(("Estructura de archivos", check_file_structure()))

    # 5. Probar consulta Ollama
    print_header("5. Probando Consulta a Ollama")
    checks.append(("Consulta Ollama", test_ollama_query()))

    # 6. Verificar recursos del sistema
    print_header("6. Verificando Recursos del Sistema")
    checks.append(("Recursos del sistema", check_system_resources()))

    # Resumen final
    print_header("RESUMEN DE VERIFICACIÓN")

    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for name, result in checks:
        if result:
            print_success(f"{name}: OK")
        else:
            print_error(f"{name}: FALLO")

    print(f"\n{Colors.BOLD}Resultado: {passed}/{total} verificaciones pasadas{Colors.RESET}\n")

    if passed == total:
        print_success("¡Sistema completamente configurado! ✓")
        print_info("\nPuedes ejecutar:")
        print("  - cd notebooks && jupyter notebook LLM_Evaluation.ipynb")
        print("  - cd scripts && python3 test_llm_models.py")
        print("  - ollama run llama3.2:1b")
        return 0
    else:
        print_error("Algunas verificaciones fallaron")
        print_info("\nConsulta README.md o QUICK_START.md para instrucciones de instalación")
        return 1

if __name__ == "__main__":
    sys.exit(main())
