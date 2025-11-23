#!/usr/bin/env python3
"""
Sistema de Pruebas y Evaluación de Modelos LLM
Parcial 5 - Implementación de Modelos LLM con Ollama
"""

import json
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple
import statistics

class LLMTester:
    """Clase para realizar pruebas y evaluación de modelos LLM"""

    def __init__(self, models: List[str]):
        """
        Inicializa el sistema de pruebas

        Args:
            models: Lista de nombres de modelos a evaluar
        """
        self.models = models
        self.results = {}
        self.test_prompts = self._get_test_prompts()

    def _get_test_prompts(self) -> List[Dict[str, str]]:
        """Define los prompts de prueba para evaluar los modelos"""
        return [
            {
                "category": "Razonamiento Lógico",
                "prompt": "Si todos los gatos tienen bigotes y Fluffy es un gato, ¿qué podemos concluir sobre Fluffy? Explica tu razonamiento paso a paso.",
                "keywords": ["bigotes", "gato", "fluffy", "tiene"]
            },
            {
                "category": "Matemáticas Básicas",
                "prompt": "Resuelve el siguiente problema paso a paso: Si tengo 15 manzanas y regalo 1/3 de ellas, ¿cuántas manzanas me quedan?",
                "keywords": ["10", "diez", "manzanas", "quedan"]
            },
            {
                "category": "Comprensión de Lectura",
                "prompt": "Lee el siguiente texto y responde: 'María estudia ingeniería en sistemas. Le gusta programar en Python.' ¿Qué le gusta hacer a María?",
                "keywords": ["programar", "python", "código"]
            },
            {
                "category": "Creatividad",
                "prompt": "Escribe un haiku sobre la tecnología.",
                "keywords": []  # Evaluación subjetiva
            },
            {
                "category": "Conocimiento General",
                "prompt": "¿Cuál es la capital de Colombia y menciona un dato interesante sobre ella?",
                "keywords": ["bogotá", "bogota", "capital"]
            },
            {
                "category": "Programación",
                "prompt": "Escribe una función en Python que calcule el factorial de un número.",
                "keywords": ["def", "factorial", "return", "recursiv"]
            },
            {
                "category": "Análisis y Síntesis",
                "prompt": "¿Cuáles son las ventajas y desventajas de usar inteligencia artificial en la educación?",
                "keywords": ["ventaja", "desventaja", "educación"]
            },
            {
                "category": "Conversación Cotidiana",
                "prompt": "¿Cómo ha sido tu día? Cuéntame algo interesante.",
                "keywords": []  # Evaluación de coherencia
            }
        ]

    def query_ollama(self, model: str, prompt: str, temperature: float = 0.7) -> Tuple[str, float, bool]:
        """
        Realiza una consulta a un modelo de Ollama

        Args:
            model: Nombre del modelo
            prompt: Pregunta o instrucción
            temperature: Parámetro de temperatura para la generación

        Returns:
            Tupla con (respuesta, tiempo_respuesta, éxito)
        """
        try:
            start_time = time.time()

            # Comando para Ollama
            cmd = [
                "ollama", "run", model,
                "--verbose"
            ]

            # Ejecutar el comando
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Enviar el prompt y obtener respuesta
            stdout, stderr = process.communicate(input=prompt + "\n/bye\n", timeout=60)

            elapsed_time = time.time() - start_time

            # Limpiar la respuesta
            response = stdout.strip()

            return response, elapsed_time, True

        except subprocess.TimeoutExpired:
            return "ERROR: Timeout", 60.0, False
        except Exception as e:
            return f"ERROR: {str(e)}", 0.0, False

    def evaluate_response(self, response: str, test_case: Dict) -> Dict[str, any]:
        """
        Evalúa la calidad de una respuesta

        Args:
            response: Respuesta del modelo
            test_case: Caso de prueba con keywords

        Returns:
            Diccionario con métricas de evaluación
        """
        evaluation = {
            "length": len(response),
            "has_content": len(response) > 10,
            "keyword_matches": 0,
            "relevance_score": 0.0
        }

        # Verificar keywords si existen
        if test_case.get("keywords"):
            response_lower = response.lower()
            matches = sum(1 for kw in test_case["keywords"] if kw.lower() in response_lower)
            evaluation["keyword_matches"] = matches
            evaluation["relevance_score"] = matches / len(test_case["keywords"]) if test_case["keywords"] else 0

        return evaluation

    def test_model(self, model: str) -> Dict:
        """
        Ejecuta todas las pruebas para un modelo específico

        Args:
            model: Nombre del modelo a evaluar

        Returns:
            Diccionario con resultados completos
        """
        print(f"\n{'='*60}")
        print(f"Evaluando modelo: {model}")
        print(f"{'='*60}\n")

        model_results = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }

        response_times = []
        relevance_scores = []
        success_count = 0

        for i, test in enumerate(self.test_prompts, 1):
            print(f"[{i}/{len(self.test_prompts)}] Categoría: {test['category']}")
            print(f"Prompt: {test['prompt'][:60]}...")

            # Realizar consulta
            response, elapsed_time, success = self.query_ollama(model, test['prompt'])

            # Evaluar respuesta
            evaluation = self.evaluate_response(response, test)

            # Guardar resultados
            test_result = {
                "category": test['category'],
                "prompt": test['prompt'],
                "response": response[:500],  # Limitar longitud para el reporte
                "response_time": elapsed_time,
                "success": success,
                "evaluation": evaluation
            }

            model_results["tests"].append(test_result)

            if success:
                response_times.append(elapsed_time)
                if evaluation["relevance_score"] > 0:
                    relevance_scores.append(evaluation["relevance_score"])
                success_count += 1

            print(f"✓ Completado en {elapsed_time:.2f}s")
            print(f"  Relevancia: {evaluation['relevance_score']*100:.1f}%\n")

            time.sleep(1)  # Pausa entre consultas

        # Calcular estadísticas resumidas
        model_results["summary"] = {
            "total_tests": len(self.test_prompts),
            "successful_tests": success_count,
            "success_rate": success_count / len(self.test_prompts) * 100,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "avg_relevance": statistics.mean(relevance_scores) if relevance_scores else 0,
            "total_time": sum(response_times)
        }

        return model_results

    def run_all_tests(self) -> Dict:
        """
        Ejecuta pruebas para todos los modelos

        Returns:
            Diccionario con todos los resultados
        """
        print("\n" + "="*60)
        print("INICIANDO EVALUACIÓN DE MODELOS LLM")
        print("="*60)
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Modelos a evaluar: {', '.join(self.models)}")
        print(f"Total de pruebas por modelo: {len(self.test_prompts)}\n")

        all_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "models_tested": self.models,
                "total_prompts": len(self.test_prompts)
            },
            "results": []
        }

        for model in self.models:
            try:
                result = self.test_model(model)
                all_results["results"].append(result)
                self.results[model] = result
            except Exception as e:
                print(f"❌ Error evaluando {model}: {str(e)}")
                all_results["results"].append({
                    "model": model,
                    "error": str(e),
                    "success": False
                })

        return all_results

    def save_results(self, filename: str = "evaluation_results.json"):
        """
        Guarda los resultados en un archivo JSON

        Args:
            filename: Nombre del archivo de salida
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Resultados guardados en: {filename}")

    def generate_comparison_report(self) -> str:
        """
        Genera un reporte comparativo de los modelos

        Returns:
            String con el reporte en formato Markdown
        """
        report = "# Reporte Comparativo de Modelos LLM\n\n"
        report += f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report += "## Resumen de Resultados\n\n"
        report += "| Modelo | Tasa Éxito | Tiempo Promedio | Tiempo Total | Relevancia Promedio |\n"
        report += "|--------|------------|----------------|--------------|--------------------|\n"

        for model, data in self.results.items():
            if "summary" in data:
                s = data["summary"]
                report += f"| {model} | {s['success_rate']:.1f}% | "
                report += f"{s['avg_response_time']:.2f}s | "
                report += f"{s['total_time']:.2f}s | "
                report += f"{s['avg_relevance']*100:.1f}% |\n"

        report += "\n## Detalles por Categoría\n\n"

        # Agrupar por categoría
        categories = {}
        for model, data in self.results.items():
            for test in data.get("tests", []):
                cat = test["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append({
                    "model": model,
                    "time": test["response_time"],
                    "relevance": test["evaluation"]["relevance_score"]
                })

        for category, tests in categories.items():
            report += f"### {category}\n\n"
            report += "| Modelo | Tiempo de Respuesta | Relevancia |\n"
            report += "|--------|-------------------|------------|\n"
            for test in tests:
                report += f"| {test['model']} | {test['time']:.2f}s | {test['relevance']*100:.1f}% |\n"
            report += "\n"

        return report


def main():
    """Función principal"""
    # Modelos a evaluar
    models = [
        "llama3.2:1b",
        "deepseek-r1:1.5b",
        "qwen2.5:0.5b"
    ]

    # Crear instancia del evaluador
    tester = LLMTester(models)

    # Ejecutar todas las pruebas
    results = tester.run_all_tests()

    # Guardar resultados
    results_dir = "../results"
    import os
    os.makedirs(results_dir, exist_ok=True)

    tester.save_results(f"{results_dir}/evaluation_results.json")

    # Generar reporte comparativo
    report = tester.generate_comparison_report()
    with open(f"{results_dir}/comparison_report.md", 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n" + "="*60)
    print("EVALUACIÓN COMPLETADA")
    print("="*60)
    print(f"\nArchivos generados:")
    print(f"  - {results_dir}/evaluation_results.json")
    print(f"  - {results_dir}/comparison_report.md")
    print("\n")


if __name__ == "__main__":
    main()
