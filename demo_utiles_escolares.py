#!/usr/bin/env python3
"""
Demo específico para el proyecto de útiles escolares
Universidad Politécnica de Chiapas

Este script demuestra el funcionamiento específico del detector
de lápices, cuadernos y kits de escritura.
"""

import cv2
import os
import time
from app import YOLOObjectDetector
import matplotlib.pyplot as plt
import json
import numpy as np

class UtilesEscolaresDemo:
    """
    Demostración específica para útiles escolares.
    """
    
    def __init__(self):
        self.detector = YOLOObjectDetector()
        print("📝 DEMO: Detector de Útiles Escolares")
        print("="*50)
        print("Clases objetivo:")
        print("🟢 Preentrenado: libros, personas, mesas (contexto escolar)")
        print("🔵 Personalizado: lápices, cuadernos, kits de escritura")
        print()
    
    def test_pretrained_school_context(self):
        """
        Prueba la detección de contexto escolar con modelo preentrenado.
        """
        print("📚 PRUEBA: Contexto Escolar (Modelo Preentrenado)")
        print("-"*40)
        
        # Información sobre las clases escolares en COCO
        school_context_classes = {
            'book': 'Libros y cuadernos grandes',
            'person': 'Estudiantes y profesores', 
            'chair': 'Sillas de estudio',
            'dining table': 'Mesas de trabajo',
            'laptop': 'Computadoras para estudio',
            'cell phone': 'Dispositivos móviles'
        }
        
        print("Clases escolares disponibles en YOLO preentrenado:")
        for class_name, description in school_context_classes.items():
            print(f"   📖 {class_name}: {description}")
        
        print("\n💡 Estas clases ayudan a establecer el contexto educativo")
        print("   donde aparecen nuestros útiles escolares personalizados.")
        print()
    
    def simulate_custom_detections(self):
        """
        Simula detecciones del modelo personalizado.
        """
        print("🎯 SIMULACIÓN: Detecciones Personalizadas")
        print("-"*40)
        
        # Simulación de resultados esperados
        simulated_results = {
            "imagen_escritorio.jpg": {
                "lapiz": 2,
                "cuaderno": 1,
                "kit_escritura": 1
            },
            "imagen_estudiante.jpg": {
                "lapiz": 1,
                "cuaderno": 1,
                "kit_escritura": 1
            },
            "imagen_salon.jpg": {
                "lapiz": 5,
                "cuaderno": 3,
                "kit_escritura": 2
            }
        }
        
        print("Resultados esperados después del entrenamiento:")
        for image_name, detections in simulated_results.items():
            print(f"\n📸 {image_name}:")
            total = sum(detections.values())
            print(f"   Total objetos detectados: {total}")
            for class_name, count in detections.items():
                emoji = "📏" if class_name == "lapiz" else "📔" if class_name == "cuaderno" else "🎒"
                print(f"   {emoji} {class_name}: {count}")
    
    def create_training_plan(self):
        """
        Crea un plan específico de entrenamiento.
        """
        print("📋 PLAN DE ENTRENAMIENTO PERSONALIZADO")
        print("-"*40)
        
        training_plan = {
            "lapiz": {
                "imagenes_objetivo": "50-75",
                "descripcion": "Lápices individuales, diferentes colores y tamaños",
                "ejemplos": [
                    "Lápiz amarillo #2 sobre mesa blanca",
                    "Lápiz mecánico en mano de estudiante", 
                    "Varios lápices de colores juntos",
                    "Lápiz gastado junto a sacapuntas"
                ]
            },
            "cuaderno": {
                "imagenes_objetivo": "50-75", 
                "descripcion": "Cuadernos cerrados y abiertos, diferentes tipos",
                "ejemplos": [
                    "Cuaderno universitario cerrado",
                    "Cuaderno espiral abierto con escritura",
                    "Stack de cuadernos apilados",
                    "Cuaderno de dibujo con lápices de colores"
                ]
            },
            "kit_escritura": {
                "imagenes_objetivo": "75-100",
                "descripcion": "Lápiz Y cuaderno juntos (CLASE MÁS IMPORTANTE)",
                "ejemplos": [
                    "Lápiz sobre cuaderno cerrado",
                    "Estudiante escribiendo con lápiz en cuaderno",
                    "Kit completo en estuche escolar",
                    "Lápiz marcando página en cuaderno abierto"
                ]
            }
        }
        
        for class_name, details in training_plan.items():
            emoji = "📏" if class_name == "lapiz" else "📔" if class_name == "cuaderno" else "🎒"
            print(f"\n{emoji} CLASE: {class_name}")
            print(f"   🎯 Objetivo: {details['imagenes_objetivo']} imágenes")
            print(f"   📝 {details['descripcion']}")
            print("   💡 Ejemplos de imágenes:")
            for example in details['ejemplos']:
                print(f"      • {example}")
    
    def estimate_performance_metrics(self):
        """
        Estima las métricas de rendimiento esperadas.
        """
        print("📊 MÉTRICAS DE RENDIMIENTO ESPERADAS")
        print("-"*40)
        
        performance_estimates = {
            "lapiz": {
                "map50": "0.75-0.85",
                "precision": "0.80-0.90", 
                "recall": "0.75-0.85",
                "dificultad": "Media (objeto alargado y fino)"
            },
            "cuaderno": {
                "map50": "0.80-0.90",
                "precision": "0.85-0.95",
                "recall": "0.80-0.90", 
                "dificultad": "Baja (objeto rectangular distintivo)"
            },
            "kit_escritura": {
                "map50": "0.70-0.80",
                "precision": "0.75-0.85",
                "recall": "0.70-0.80",
                "dificultad": "Alta (concepto compuesto, requiere contexto)"
            }
        }
        
        print("Rendimiento estimado por clase:")
        for class_name, metrics in performance_estimates.items():
            emoji = "📏" if class_name == "lapiz" else "📔" if class_name == "cuaderno" else "🎒"
            print(f"\n{emoji} {class_name.upper()}:")
            print(f"   mAP50: {metrics['map50']}")
            print(f"   Precisión: {metrics['precision']}")
            print(f"   Recall: {metrics['recall']}")
            print(f"   Dificultad: {metrics['dificultad']}")
        
        print(f"\n🎯 OBJETIVO GENERAL DEL PROYECTO:")
        print(f"   mAP50 promedio > 0.75")
        print(f"   Precisión promedio > 0.80") 
        print(f"   Recall promedio > 0.75")
    
    def create_dataset_examples(self):
        """
        Crea ejemplos específicos para el dataset.
        """
        print("📷 EJEMPLOS ESPECÍFICOS PARA DATASET")
        print("-"*40)
        
        dataset_scenarios = {
            "Escenario 1: Escritorio de estudiante": [
                "📏 Lápiz sobre mesa + 📔 cuaderno abierto = 🎒 kit_escritura",
                "Contexto: mesa de madera, buena iluminación",
                "Ángulo: vista superior (como estudiante viendo su escritorio)"
            ],
            "Escenario 2: Salón de clases": [
                "👥 Varios estudiantes con sus útiles",
                "📏 Múltiples lápices + 📔 múltiples cuadernos", 
                "🎒 Varios kits de escritura simultáneos"
            ],
            "Escenario 3: Biblioteca": [
                "📚 Libros (detectados por modelo preentrenado)",
                "📏 Lápiz para tomar notas + 📔 cuaderno de apuntes",
                "Contexto académico completo"
            ],
            "Escenario 4: Estuche escolar": [
                "🎒 Kit organizado en estuche",
                "📏 Lápices ordenados + 📔 cuaderno pequeño",
                "Vista de organización estudiantil"
            ]
        }
        
        for scenario, details in dataset_scenarios.items():
            print(f"\n🎬 {scenario}:")
            for detail in details:
                print(f"   • {detail}")
    
    def run_complete_demo(self):
        """
        Ejecuta la demostración completa específica para útiles escolares.
        """
        print("🎯 DEMOSTRACIÓN COMPLETA: ÚTILES ESCOLARES")
        print("Universidad Politécnica de Chiapas")
        print("="*60)
        print()
        
        sections = [
            ("Contexto Escolar (Preentrenado)", self.test_pretrained_school_context),
            ("Detecciones Personalizadas", self.simulate_custom_detections), 
            ("Plan de Entrenamiento", self.create_training_plan),
            ("Métricas Esperadas", self.estimate_performance_metrics),
            ("Ejemplos de Dataset", self.create_dataset_examples)
        ]
        
        for section_name, section_method in sections:
            print(f"\n{'='*20} {section_name} {'='*20}")
            section_method()
            
            if section_name != "Ejemplos de Dataset":  # No pausa en la última sección
                input(f"\n⏸️  Presiona Enter para continuar a '{sections[sections.index((section_name, section_method)) + 1][0] if sections.index((section_name, section_method)) < len(sections) - 1 else 'Finalizar'}'...")
        
        print("\n" + "="*60)
        print("🎉 DEMOSTRACIÓN COMPLETADA")
        print("="*60)
        print()
        print("📋 PRÓXIMOS PASOS PARA TU PROYECTO:")
        print("1. 📸 Recolectar imágenes según el plan mostrado")
        print("2. 🏷️ Anotar con LabelImg o Roboflow")
        print("3. 🚀 Entrenar modelo: python train_custom_model.py")
        print("4. 🎯 Probar aplicación: python app.py")
        print("5. 📊 Documentar resultados y métricas")
        print()
        print("💡 CONSEJO CLAVE:")
        print("   La clase 'kit_escritura' es tu diferenciador único.")
        print("   Asegúrate de tener ejemplos variados y bien anotados.")

def main():
    """
    Función principal para la demostración de útiles escolares.
    """
    demo = UtilesEscolaresDemo()
    
    print("📝 OPCIONES DE DEMOSTRACIÓN:")
    print("1. Demostración completa (recomendado)")
    print("2. Solo plan de entrenamiento")
    print("3. Solo métricas esperadas")
    print("4. Solo ejemplos de dataset")
    
    try:
        choice = input("\nElige una opción (1-4): ").strip()
        print()
        
        if choice == "1":
            demo.run_complete_demo()
        elif choice == "2":
            demo.create_training_plan()
        elif choice == "3":
            demo.estimate_performance_metrics()
        elif choice == "4":
            demo.create_dataset_examples()
        else:
            print("Opción inválida. Ejecutando demostración completa...")
            demo.run_complete_demo()
            
    except KeyboardInterrupt:
        print("\n👋 Demostración interrumpida")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()