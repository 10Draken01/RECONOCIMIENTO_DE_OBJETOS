#!/usr/bin/env python3
"""
Script para entrenar un modelo YOLO personalizado
Universidad Politécnica de Chiapas

Este script:
1. Prepara el dataset personalizado
2. Configura el entrenamiento
3. Entrena el modelo YOLO
4. Genera métricas y visualizaciones
5. Valida el modelo entrenado

Para usar este script necesitas:
- Imágenes anotadas en formato YOLO
- Archivo data.yaml con configuración del dataset
"""

import os
import yaml
import shutil
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

class CustomYOLOTrainer:
    """
    Clase para entrenar modelos YOLO personalizados.
    
    Maneja todo el proceso desde la preparación del dataset
    hasta la validación del modelo entrenado.
    """
    
    def __init__(self, project_name: str = "custom_detection"):
        """
        Inicializa el entrenador personalizado.
        
        Args:
            project_name: Nombre del proyecto de entrenamiento
        """
        self.project_name = project_name
        self.dataset_path = "dataset"
        self.models_path = "models"
        self.results_path = "training_results"
        
        # Crear directorios necesarios
        self.create_directories()
        
        # Configuración de entrenamiento
        self.training_config = {
            'epochs': 100,          # Número de epochs (ciclos de entrenamiento)
            'imgsz': 640,          # Tamaño de imagen para entrenamiento
            'batch': 16,           # Tamaño del batch (ajustar según tu GPU)
            'lr0': 0.01,           # Learning rate inicial
            'patience': 50,        # Paciencia para early stopping
            'save_period': 10,     # Guardar modelo cada X epochs
        }
    
    def create_directories(self):
        """
        Crea la estructura de directorios necesaria.
        """
        directories = [
            self.dataset_path,
            f"{self.dataset_path}/images/train",
            f"{self.dataset_path}/images/val",
            f"{self.dataset_path}/labels/train",
            f"{self.dataset_path}/labels/val",
            self.models_path,
            self.results_path
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Directorio creado/verificado: {directory}")
    
    def create_dataset_yaml(self, class_names: list, class_descriptions: list = None):
        """
        Crea el archivo data.yaml necesario para YOLO.
        
        Args:
            class_names: Lista con nombres de las clases a detectar
            class_descriptions: Descripciones opcionales de las clases
        """
        if class_descriptions is None:
            class_descriptions = [f"Descripción de {name}" for name in class_names]
        
        # Configuración del dataset
        dataset_config = {
            'path': os.path.abspath(self.dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_names),  # Número de clases
            'names': class_names
        }
        
        # Guardar archivo YAML
        yaml_path = f"{self.dataset_path}/data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Archivo data.yaml creado en: {yaml_path}")
        print(f"📊 Configuración: {len(class_names)} clases")
        for i, (name, desc) in enumerate(zip(class_names, class_descriptions)):
            print(f"   Clase {i}: {name} - {desc}")
        
        return yaml_path
    
    def validate_dataset(self):
        """
        Valida que el dataset tenga la estructura correcta.
        
        Returns:
            bool: True si el dataset es válido
        """
        print("🔍 Validando estructura del dataset...")
        
        # Verificar que existan las carpetas necesarias
        required_dirs = [
            f"{self.dataset_path}/images/train",
            f"{self.dataset_path}/images/val",
            f"{self.dataset_path}/labels/train",
            f"{self.dataset_path}/labels/val"
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                print(f"❌ Directorio faltante: {directory}")
                return False
        
        # Contar imágenes y etiquetas
        train_images = len([f for f in os.listdir(f"{self.dataset_path}/images/train") 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_images = len([f for f in os.listdir(f"{self.dataset_path}/images/val") 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        train_labels = len([f for f in os.listdir(f"{self.dataset_path}/labels/train") 
                           if f.endswith('.txt')])
        val_labels = len([f for f in os.listdir(f"{self.dataset_path}/labels/val") 
                         if f.endswith('.txt')])
        
        print(f"📊 Estadísticas del dataset:")
        print(f"   Imágenes de entrenamiento: {train_images}")
        print(f"   Etiquetas de entrenamiento: {train_labels}")
        print(f"   Imágenes de validación: {val_images}")
        print(f"   Etiquetas de validación: {val_labels}")
        
        # Verificar que haya suficientes datos
        if train_images < 50:
            print("⚠️ ADVERTENCIA: Pocas imágenes de entrenamiento (mínimo recomendado: 50)")
        
        if train_images != train_labels:
            print("⚠️ ADVERTENCIA: Número de imágenes y etiquetas no coincide en entrenamiento")
        
        # Verificar archivo data.yaml
        yaml_path = f"{self.dataset_path}/data.yaml"
        if not os.path.exists(yaml_path):
            print(f"❌ Archivo data.yaml faltante: {yaml_path}")
            return False
        
        print("✅ Dataset validado correctamente")
        return True
    
    def train_model(self, base_model: str = "yolov8n.pt"):
        """
        Entrena el modelo YOLO personalizado.
        
        Args:
            base_model: Modelo base para transfer learning
        """
        if not self.validate_dataset():
            print("❌ Dataset inválido. Por favor corrige los errores antes de entrenar.")
            return None
        
        print(f"🚀 Iniciando entrenamiento con modelo base: {base_model}")
        
        # Cargar modelo base
        model = YOLO(base_model)
        
        # Configurar paths
        data_yaml = f"{self.dataset_path}/data.yaml"
        project_path = self.results_path
        
        print("📋 Configuración de entrenamiento:")
        for key, value in self.training_config.items():
            print(f"   {key}: {value}")
        
        try:
            # Iniciar entrenamiento
            results = model.train(
                data=data_yaml,
                epochs=self.training_config['epochs'],
                imgsz=self.training_config['imgsz'],
                batch=self.training_config['batch'],
                lr0=self.training_config['lr0'],
                patience=self.training_config['patience'],
                save_period=self.training_config['save_period'],
                project=project_path,
                name=self.project_name,
                exist_ok=True,
                verbose=True
            )
            
            print("✅ Entrenamiento completado exitosamente")
            
            # Copiar mejor modelo a carpeta models
            best_model_path = f"{project_path}/{self.project_name}/weights/best.pt"
            final_model_path = f"{self.models_path}/custom_model.pt"
            
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, final_model_path)
                print(f"✅ Mejor modelo guardado en: {final_model_path}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {str(e)}")
            return None
    
    def validate_trained_model(self):
        """
        Valida el modelo entrenado y genera métricas.
        """
        model_path = f"{self.models_path}/custom_model.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ Modelo no encontrado: {model_path}")
            return
        
        print("🔍 Validando modelo entrenado...")
        
        # Cargar modelo entrenado
        model = YOLO(model_path)
        
        # Ejecutar validación
        data_yaml = f"{self.dataset_path}/data.yaml"
        validation_results = model.val(data=data_yaml)
        
        # Generar reporte de métricas
        self.generate_metrics_report(validation_results)
        
        print("✅ Validación completada")
    
    def generate_metrics_report(self, validation_results):
        """
        Genera un reporte detallado de las métricas del modelo.
        
        Args:
            validation_results: Resultados de la validación de YOLO
        """
        print("📊 Generando reporte de métricas...")
        
        # Crear reporte en texto
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.results_path}/metrics_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE MÉTRICAS - MODELO YOLO PERSONALIZADO\n")
            f.write("="*60 + "\n\n")
            f.write(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Proyecto: {self.project_name}\n\n")
            
            # Métricas principales
            if hasattr(validation_results, 'box'):
                f.write("MÉTRICAS DE DETECCIÓN:\n")
                f.write(f"mAP50: {validation_results.box.map50:.4f}\n")
                f.write(f"mAP50-95: {validation_results.box.map:.4f}\n")
                f.write(f"Precisión: {validation_results.box.mp:.4f}\n")
                f.write(f"Recall: {validation_results.box.mr:.4f}\n\n")
            
            f.write("CONFIGURACIÓN DE ENTRENAMIENTO:\n")
            for key, value in self.training_config.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✅ Reporte guardado en: {report_path}")
    
    def create_training_guide(self):
        """
        Crea una guía paso a paso para preparar el dataset.
        """
        guide_path = f"{self.results_path}/guia_dataset.md"
        
        guide_content = """
# 📋 Guía para Preparar Dataset Personalizado

## 🎯 Objetivo
Crear un dataset para entrenar YOLO a detectar objetos específicos (ej: logos, productos, herramientas).

## 📁 Estructura de Carpetas Requerida
```
dataset/
├── data.yaml           # Configuración del dataset
├── images/
│   ├── train/         # Imágenes de entrenamiento (80%)
│   └── val/           # Imágenes de validación (20%)
└── labels/
    ├── train/         # Etiquetas de entrenamiento (.txt)
    └── val/           # Etiquetas de validación (.txt)
```

## 📷 Paso 1: Recolectar Imágenes
1. **Cantidad mínima**: 50 imágenes por clase
2. **Recomendado**: 100-300 imágenes por clase
3. **Variedad**: Diferentes ángulos, iluminación, fondos
4. **Formato**: JPG, JPEG o PNG
5. **Resolución**: Mínimo 640x640 píxeles

### Consejos para mejores resultados:
- Incluye objetos en diferentes posiciones
- Varía la iluminación (natural, artificial)
- Diferentes fondos y contextos
- Objetos parcialmente ocluidos
- Diferentes escalas (cerca, lejos)

## 🏷️ Paso 2: Anotar Imágenes

### Opción A: Usar LabelImg (Recomendado)
1. Instalar: `pip install labelimg`
2. Ejecutar: `labelimg`
3. Abrir directorio de imágenes
4. Cambiar formato a "YOLO"
5. Crear cajas delimitadoras alrededor de objetos
6. Guardar etiquetas (.txt)

### Opción B: Usar Roboflow (Online)
1. Ir a https://roboflow.com
2. Crear proyecto gratuito
3. Subir imágenes
4. Anotar online
5. Exportar en formato YOLO

## 📊 Paso 3: Dividir Dataset
- **Entrenamiento**: 80% de las imágenes
- **Validación**: 20% de las imágenes
- Distribuir equitativamente por clase

## 🔧 Paso 4: Crear data.yaml
Usar la función `create_dataset_yaml()` del script.

## 🚀 Paso 5: Entrenar Modelo
Ejecutar el script de entrenamiento.

## 📈 Métricas a Observar
- **mAP50**: Precisión promedio a IoU=0.5
- **mAP50-95**: Precisión promedio IoU=0.5-0.95
- **Precisión**: % de detecciones correctas
- **Recall**: % de objetos reales detectados

### Valores objetivo:
- mAP50 > 0.7 (Bueno)
- mAP50 > 0.8 (Excelente)
- Precisión > 0.8
- Recall > 0.7
"""
        
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"📋 Guía creada en: {guide_path}")

def main():
    """
    Función principal para entrenar modelo personalizado.
    """
    print("🎯 ENTRENADOR DE MODELO YOLO PERSONALIZADO")
    print("="*50)
    
    # Crear instancia del entrenador
    trainer = CustomYOLOTrainer("mi_detector_personalizado")
    
    # Generar guía
    trainer.create_training_guide()
    
    # Ejemplo de uso - PERSONALIZAR SEGÚN TU CASO
    print("\n📋 CONFIGURACIÓN DE EJEMPLO:")
    print("Para entrenar tu modelo, sigue estos pasos:")
    print("1. Define las clases que quieres detectar")
    print("2. Prepara tu dataset siguiendo la guía generada")
    print("3. Ejecuta las funciones de entrenamiento")
    
    # Ejemplo para detectar útiles escolares
    example_classes = ["lapiz", "cuaderno", "kit_escritura"]
    example_descriptions = [
        "Lápiz - instrumento de escritura",
        "Cuaderno - libreta para escribir o dibujar", 
        "Kit de escritura - lápiz y cuaderno juntos"
    ]
    
    print(f"\n🎯 EJEMPLO: Detector de útiles escolares")
    print(f"Clases: {example_classes}")
    
    # Crear archivo data.yaml de ejemplo
    yaml_path = trainer.create_dataset_yaml(example_classes, example_descriptions)
    
    print(f"\n📋 PRÓXIMOS PASOS:")
    print(f"1. Coloca tus imágenes anotadas en: dataset/images/train y dataset/images/val")
    print(f"2. Coloca las etiquetas (.txt) en: dataset/labels/train y dataset/labels/val")
    print(f"3. Ejecuta: trainer.train_model()")
    print(f"4. Valida con: trainer.validate_trained_model()")
    
    # Preguntar si desea iniciar entrenamiento
    response = input("\n¿Tienes tu dataset listo y quieres iniciar el entrenamiento? (s/n): ")
    if response.lower() == 's':
        print("🚀 Iniciando entrenamiento...")
        results = trainer.train_model()
        if results:
            trainer.validate_trained_model()
    else:
        print("📋 Prepara tu dataset y vuelve a ejecutar este script.")

if __name__ == "__main__":
    main()