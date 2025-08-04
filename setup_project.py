#!/usr/bin/env python3
"""
Script de configuración inicial para el proyecto YOLO
Universidad Politécnica de Chiapas

Este script:
1. Instala dependencias automáticamente
2. Descarga modelos preentrenados
3. Crea estructura de carpetas
4. Proporciona ejemplos de uso
5. Verifica que todo funcione correctamente
"""

import subprocess
import sys
import os
import urllib.request
import zipfile
from pathlib import Path

def install_requirements():
    """
    Instala todas las dependencias necesarias.
    """
    print("📦 Instalando dependencias...")
    
    requirements = [
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0", 
        "gradio>=3.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "requests>=2.25.0"
    ]
    
    for package in requirements:
        try:
            print(f"   Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"   ✅ {package} instalado")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error instalando {package}: {e}")

def create_project_structure():
    """
    Crea la estructura completa de carpetas del proyecto.
    """
    print("📁 Creando estructura de carpetas...")
    
    folders = [
        "dataset/images/train",
        "dataset/images/val", 
        "dataset/labels/train",
        "dataset/labels/val",
        "models",
        "training_results",
        "test_images",
        "exports",
        "documentation"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"   📁 {folder}/")
    
    print("✅ Estructura de carpetas creada")

def download_sample_images():
    """
    Descarga imágenes de ejemplo para probar la aplicación.
    """
    print("🖼️ Descargando imágenes de ejemplo...")
    
    # URLs de imágenes de ejemplo (usar imágenes libres de derechos)
    sample_urls = [
        "https://images.unsplash.com/photo-1544966503-7cc1d85d6a4b?w=640",  # Personas
        "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640",  # Coches
        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=640",   # Bicicletas
    ]
    
    for i, url in enumerate(sample_urls):
        try:
            filename = f"test_images/ejemplo_{i+1}.jpg"
            print(f"   Descargando {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"   ✅ {filename} descargado")
        except Exception as e:
            print(f"   ⚠️ No se pudo descargar imagen {i+1}: {e}")

def create_documentation():
    """
    Crea documentación del proyecto.
    """
    print("📝 Creando documentación...")
    
    readme_content = """
# 🎯 Detector de Objetos YOLO - UPChiapas

## 📋 Descripción del Proyecto
Aplicación de reconocimiento de objetos que combina:
- **Modelo YOLO preentrenado**: Detecta personas, coches, bicicletas, etc.
- **Modelo YOLO personalizado**: Detecta objetos específicos entrenados por nosotros
- **Interfaz web con Gradio**: Fácil de usar y visualmente atractiva

## 🚀 Instalación y Uso

### 1. Configuración Inicial
```bash
python setup_project.py
```

### 2. Entrenar Modelo Personalizado
```bash
python train_custom_model.py
```

### 3. Ejecutar Aplicación
```bash
python app.py
```

## 📁 Estructura del Proyecto
```
proyecto_yolo/
├── app.py                     # Aplicación principal
├── train_custom_model.py      # Entrenamiento personalizado
├── setup_project.py           # Configuración inicial
├── requirements.txt           # Dependencias
├── dataset/                   # Dataset personalizado
│   ├── data.yaml
│   ├── images/
│   └── labels/
├── models/                    # Modelos entrenados
├── test_images/              # Imágenes de prueba
└── documentation/            # Documentación

```

## 🎯 Funcionalidades Principales
- ✅ Detección con modelo preentrenado (80 clases COCO)
- ✅ Detección con modelo personalizado
- ✅ Interfaz web interactiva
- ✅ Procesamiento de imágenes y video
- ✅ Conteo automático de objetos
- ✅ Exportación de resultados (CSV/JSON)
- ✅ Visualización con cajas delimitadoras

## 📊 Métricas del Modelo
[Se actualizarán después del entrenamiento]

## 👥 Desarrolladores
- [Tu Nombre]
- [Nombre del Compañero]

**Universidad**: Politécnica de Chiapas
**Materia**: Multimedia y Diseño Digital
**Profesor**: [Nombre del Profesor]
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Crear archivo de configuración
    config_content = """
# Configuración del Proyecto YOLO

## Modelos Disponibles
- yolov8n.pt: Modelo nano (más rápido)
- yolov8s.pt: Modelo small (balance)  
- yolov8m.pt: Modelo medium (más preciso)
- yolov8l.pt: Modelo large (muy preciso)
- yolov8x.pt: Modelo extra large (máxima precisión)

## Clases del Modelo Preentrenado (COCO)
0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 5: bus,
6: train, 7: truck, 8: boat, 9: traffic light, 10: fire hydrant,
[... 80 clases total]

## Configuración de Entrenamiento Recomendada
- Epochs: 100-300 (según dataset)
- Batch size: 16 (ajustar según GPU)
- Image size: 640x640
- Learning rate: 0.01
"""
    
    with open("documentation/configuracion.md", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ Documentación creada")

def test_installation():
    """
    Prueba que todas las dependencias estén instaladas correctamente.
    """
    print("🧪 Probando instalación...")
    
    try:
        # Probar imports principales
        import cv2
        print("   ✅ OpenCV disponible")
        
        import gradio as gr
        print("   ✅ Gradio disponible")
        
        from ultralytics import YOLO
        print("   ✅ Ultralytics YOLO disponible")
        
        import torch
        print(f"   ✅ PyTorch disponible (GPU: {'Sí' if torch.cuda.is_available() else 'No'})")
        
        # Probar carga de modelo YOLO
        model = YOLO('yolov8n.pt')
        print("   ✅ Modelo YOLO cargado correctamente")
        
        print("🎉 ¡Instalación exitosa! Todo está listo.")
        return True
        
    except ImportError as e:
        print(f"   ❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error general: {e}")
        return False

def create_example_usage():
    """
    Crea ejemplos de uso del código.
    """
    print("📝 Creando ejemplos de uso...")
    
    example_code = '''
# Ejemplo 1: Uso básico de la aplicación
from app import YOLOObjectDetector
import cv2

# Crear detector
detector = YOLOObjectDetector()

# Cargar imagen
image = cv2.imread("test_images/ejemplo_1.jpg")

# Procesar imagen
result_image, results_json, stats = detector.process_image(image)

# Mostrar resultados
print(stats)

# Ejemplo 2: Entrenar modelo personalizado
from train_custom_model import CustomYOLOTrainer

# Crear entrenador
trainer = CustomYOLOTrainer("mi_detector")

# Definir clases personalizadas
classes = ["logo_upchiapas", "producto_especial"]
trainer.create_dataset_yaml(classes)

# Entrenar (después de preparar dataset)
# trainer.train_model()

# Ejemplo 3: Usar solo modelo preentrenado
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model('test_images/ejemplo_1.jpg')

for result in results:
    result.show()  # Mostrar imagen con detecciones
'''
    
    with open("documentation/ejemplos.py", "w", encoding="utf-8") as f:
        f.write(example_code)
    
    print("✅ Ejemplos creados en documentation/ejemplos.py")

def main():
    """
    Configuración completa del proyecto.
    """
    print("🎯 CONFIGURACIÓN DEL PROYECTO YOLO - UPChiapas")
    print("="*60)
    print("Este script configurará todo lo necesario para tu proyecto.")
    print()
    
    # Paso 1: Instalar dependencias
    install_requirements()
    print()
    
    # Paso 2: Crear estructura
    create_project_structure()
    print()
    
    # Paso 3: Descargar ejemplos
    download_sample_images()
    print()
    
    # Paso 4: Crear documentación
    create_documentation()
    print()
    
    # Paso 5: Crear ejemplos
    create_example_usage()
    print()
    
    # Paso 6: Probar instalación
    if test_installation():
        print("\n🎉 ¡CONFIGURACIÓN COMPLETADA!")
        print("="*60)
        print("Próximos pasos:")
        print("1. 📸 Prepara tu dataset personalizado (mínimo 50 imágenes)")
        print("2. 🏷️ Anota las imágenes con LabelImg o Roboflow")
        print("3. 🚀 Ejecuta: python train_custom_model.py")
        print("4. 🎯 Ejecuta: python app.py")
        print("5. 🌐 Abre http://localhost:7860 en tu navegador")
        print()
        print("📚 Lee README.md para más información")
    else:
        print("\n❌ Hubo problemas en la configuración.")
        print("Por favor revisa los errores e intenta nuevamente.")

if __name__ == "__main__":
    main()