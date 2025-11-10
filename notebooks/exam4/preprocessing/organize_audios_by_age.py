#!/usr/bin/env python3
"""
Script para organizar archivos de audio por categoría de edad.

Este script lee el archivo train.tsv de Common Voice y organiza los archivos
de audio en carpetas según la categoría de edad de los hablantes.
"""

import os
import csv
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def create_age_folders(base_path, age_categories):
    """
    Crea las carpetas para cada categoría de edad.

    Args:
        base_path: Ruta base donde se crearán las carpetas
        age_categories: Lista de categorías de edad
    """
    created_folders = []
    for category in age_categories:
        folder_name = category if category else "no_age"
        folder_path = os.path.join(base_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        created_folders.append(folder_path)
        print(f"✓ Carpeta creada: {folder_path}")

    return created_folders


def organize_audios(tsv_path, clips_path, output_base_path):
    """
    Organiza los archivos de audio según la categoría de edad.

    Args:
        tsv_path: Ruta al archivo train.tsv
        clips_path: Ruta a la carpeta con los clips de audio
        output_base_path: Ruta base donde se crearán las carpetas organizadas
    """
    print(f"\n{'='*60}")
    print(f"Organizando audios por categoría de edad")
    print(f"{'='*60}\n")

    # Verificar que los archivos/carpetas existen
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"No se encontró el archivo TSV: {tsv_path}")

    if not os.path.exists(clips_path):
        raise FileNotFoundError(f"No se encontró la carpeta de clips: {clips_path}")

    # Leer el archivo TSV y recopilar información
    print("📖 Leyendo archivo TSV...")
    audio_mapping = defaultdict(list)
    age_stats = defaultdict(int)

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            age = row.get('age', '').strip()
            audio_file = row.get('path', '').strip()

            if not audio_file:
                continue

            # Usar "no_age" para valores vacíos
            age_category = age if age else "no_age"
            audio_mapping[age_category].append(audio_file)
            age_stats[age_category] += 1

    # Mostrar estadísticas
    print(f"\n📊 Estadísticas de categorías de edad:")
    print(f"{'-'*60}")
    for age, count in sorted(age_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {age:15} : {count:6} archivos")
    print(f"{'-'*60}")
    print(f"  {'TOTAL':15} : {sum(age_stats.values()):6} archivos\n")

    # Crear carpetas para cada categoría
    print("📁 Creando carpetas de categorías...")
    age_categories = list(audio_mapping.keys())
    create_age_folders(output_base_path, age_categories)

    # Copiar archivos a sus respectivas carpetas
    print(f"\n📋 Copiando archivos de audio...")
    stats = {
        'copied': 0,
        'not_found': 0,
        'errors': 0
    }

    missing_files = []

    for age_category, audio_files in audio_mapping.items():
        folder_name = age_category if age_category else "no_age"
        dest_folder = os.path.join(output_base_path, folder_name)

        # Usar tqdm para mostrar progreso por categoría
        print(f"\n  Procesando categoría: {age_category}")
        for audio_file in tqdm(audio_files, desc=f"  {folder_name}", unit="archivo"):
            source_path = os.path.join(clips_path, audio_file)
            dest_path = os.path.join(dest_folder, audio_file)

            try:
                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                    stats['copied'] += 1
                else:
                    stats['not_found'] += 1
                    missing_files.append(audio_file)
            except Exception as e:
                stats['errors'] += 1
                print(f"\n  ⚠️  Error al copiar {audio_file}: {e}")

    # Mostrar resultados finales
    print(f"\n{'='*60}")
    print(f"✅ Proceso completado")
    print(f"{'='*60}")
    print(f"  Archivos copiados exitosamente: {stats['copied']}")
    print(f"  Archivos no encontrados: {stats['not_found']}")
    print(f"  Errores durante la copia: {stats['errors']}")
    print(f"{'='*60}\n")

    # Guardar lista de archivos no encontrados si existen
    if missing_files:
        missing_file_path = os.path.join(output_base_path, "missing_files.txt")
        with open(missing_file_path, 'w', encoding='utf-8') as f:
            for file in missing_files:
                f.write(f"{file}\n")
        print(f"⚠️  Lista de archivos no encontrados guardada en: {missing_file_path}\n")


def main():
    """Función principal del script."""
    # Definir rutas (ajustar según la estructura del proyecto)
    base_dir = Path(__file__).parent.parent

    tsv_path = base_dir / "audios" / "cv-corpus-23.0-2025-09-05" / "pt" / "train.tsv"
    clips_path = base_dir / "audios" / "cv-corpus-23.0-2025-09-05" / "pt" / "clips"
    output_base_path = base_dir / "audios" / "organized_by_age"

    # Convertir a strings
    tsv_path = str(tsv_path)
    clips_path = str(clips_path)
    output_base_path = str(output_base_path)

    print(f"\n🔧 Configuración:")
    print(f"  - Archivo TSV: {tsv_path}")
    print(f"  - Carpeta de clips: {clips_path}")
    print(f"  - Carpeta de salida: {output_base_path}")

    try:
        organize_audios(tsv_path, clips_path, output_base_path)
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
