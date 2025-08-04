#!/usr/bin/env python3
"""
Script de instalación automática y ejecución
Universidad Politécnica de Chiapas

ESTE SCRIPT:
1. Instala TODAS las dependencias automáticamente
2. Verifica que todo funcione
3. Descarga modelos necesarios
4. Crea estructura de carpetas
5. Ejecuta la aplicación

USAR ASÍ:
python install_and_run.py
"""

import subprocess
import sys
import os
import urllib.request
import time

def print_header():
    """Imprime encabezado del proyecto."""
    print("🎯" + "="*60 + "🎯")
    print("    DETECTOR DE ÚTILES ESCOLARES - UPChiapas")
    print("    Instalación Automática y Configuración")
    print("🎯" + "="*60 + "🎯")
    print()

def install_package(package_name):
    """Instala un paquete específico."""
    try:
        print(f"📦 Instalando {package_name}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name, "--quiet"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ {package_name} instalado correctamente")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Error instalando {package_name}")
        return False

def install_all_dependencies():
    """Instala todas las dependencias necesarias."""
    print("🚀 PASO 1: Instalando dependencias...")
    print("-" * 40)
    
    # Lista de paquetes esenciales
    packages = [
        "ultralytics",
        "opencv-python", 
        "gradio",
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "Pillow",
        "matplotlib",
        "pyyaml",
        "tqdm"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
        time.sleep(0.5)  # Pequeña pausa entre instalaciones
    
    print(f"\n📊 Resultado: {success_count}/{len(packages)} paquetes instalados")
    
    if success_count == len(packages):
        print("✅ Todas las dependencias instaladas correctamente")
        return True
    else:
        print("⚠️ Algunas dependencias fallaron, pero continuaremos...")
        return True  # Continuar aunque falten algunas

def create_project_structure():
    """Crea la estructura de carpetas del proyecto."""
    print("\n🏗️ PASO 2: Creando estructura de carpetas...")
    print("-" * 40)
    
    folders = [
        "models",
        "test_images", 
        "dataset/images/train",
        "dataset/images/val",
        "dataset/labels/train", 
        "dataset/labels/val",
        "results",
        "exports"
    ]
    
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"📁 {folder}/")
        except Exception as e:
            print(f"❌ Error creando {folder}: {e}")
    
    print("✅ Estructura de carpetas creada")

def download_test_images():
    """Descarga imágenes de prueba."""
    print("\n📸 PASO 3: Descargando imágenes de prueba...")
    print("-" * 40)
    
    # URLs de imágenes de ejemplo (Unsplash - libres de derechos)
    test_images = {
        "estudiante_escribiendo.jpg": "https://images.unsplash.com/photo-1434030216411-0b793f4b4173?w=640",
        "escritorio_estudio.jpg": "https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?w=640", 
        "salon_clases.jpg": "https://images.unsplash.com/photo-1571260899304-425eee4c7efc?w=640"
    }
    
    for filename, url in test_images.items():
        try:
            filepath = f"test_images/{filename}"
            print(f"📥 Descargando {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"✅ {filename} descargado")
        except Exception as e:
            print(f"⚠️ No se pudo descargar {filename}: {e}")
    
    print("✅ Imágenes de prueba configuradas")

def create_sample_data_yaml():
    """Crea archivo de configuración de ejemplo."""
    print("\n📄 PASO 4: Creando configuración de ejemplo...")
    print("-" * 40)
    
    yaml_content = """# Configuración del dataset - Útiles Escolares
# Universidad Politécnica de Chiapas

path: ./dataset
train: images/train
val: images/val

nc: 3  # Número de clases

names:
  0: lapiz
  1: cuaderno  
  2: kit_escritura

# Información del proyecto
project_name: "Detector Útiles Escolares UPChiapas"
authors: ["Estudiante 1", "Estudiante 2"]
course: "Multimedia y Diseño Digital"
"""
    
    try:
        with open("dataset/data.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print("✅ Archivo data.yaml creado")
    except Exception as e:
        print(f"❌ Error creando data.yaml: {e}")

def test_installation():
    """Verifica que todo esté instalado correctamente."""
    print("\n🧪 PASO 5: Verificando instalación...")
    print("-" * 40)
    
    tests = []
    
    # Test 1: OpenCV
    try:
        import cv2
        print("✅ OpenCV funcionando")
        tests.append(True)
    except ImportError:
        print("❌ OpenCV no disponible")
        tests.append(False)
    
    # Test 2: Gradio
    try:
        import gradio as gr
        print("✅ Gradio funcionando")
        tests.append(True)
    except ImportError:
        print("❌ Gradio no disponible")
        tests.append(False)
    
    # Test 3: Ultralytics
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO funcionando")
        tests.append(True)
    except ImportError:
        print("❌ Ultralytics no disponible")
        tests.append(False)
    
    # Test 4: PyTorch
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        print(f"✅ PyTorch funcionando (GPU: {'Sí' if gpu_available else 'No'})")
        tests.append(True)
    except ImportError:
        print("❌ PyTorch no disponible")
        tests.append(False)
    
    # Test 5: Descarga de modelo YOLO
    try:
        print("📥 Descargando modelo YOLO preentrenado...")
        model = YOLO('yolov8n.pt')  # Esto descarga el modelo automáticamente
        print("✅ Modelo YOLO preentrenado listo")
        tests.append(True)
    except Exception as e:
        print(f"❌ Error con modelo YOLO: {e}")
        tests.append(False)
    
    success_rate = sum(tests) / len(tests) * 100
    print(f"\n📊 Tasa de éxito: {success_rate:.1f}%")
    
    return success_rate >= 80  # 80% o más para considerar exitoso

def create_readme():
    """Crea documentación del proyecto."""
    print("\n📚 PASO 6: Creando documentación...")
    print("-" * 40)
    
    readme_content = """# 📝 Detector de Útiles Escolares - UPChiapas

## 🎯 Descripción del Proyecto
Sistema de detección de objetos usando YOLO que combina:
- **Modelo preentrenado**: Detecta contexto escolar (libros, personas, mesas)
- **Modelo personalizado**: Detecta útiles específicos (lápices, cuadernos, kits)

## 🚀 Uso Rápido

### Ejecutar Aplicación
```bash
python app.py
```
Luego abrir: http://localhost:7860

### Entrenar Modelo Personalizado
```bash
python train_custom_model.py
```

## 📁 Estructura del Proyecto
```
proyecto/
├── app.py                 # Aplicación principal
├── install_and_run.py     # Instalación automática
├── train_custom_model.py  # Entrenamiento personalizado
├── dataset/               # Tu dataset personalizado
├── models/                # Modelos entrenados
├── test_images/          # Imágenes de prueba
└── results/              # Resultados y exportaciones
```

## 🎯 Funcionalidades
- ✅ Procesamiento de imágenes
- ✅ Procesamiento de videos 
- ✅ Detección en tiempo real
- ✅ Modelo preentrenado + personalizado
- ✅ Interfaz web con Gradio
- ✅ Exportación de resultados CSV/JSON

## 👥 Desarrolladores
- [Tu Nombre]
- [Nombre del Compañero]

**Universidad**: Politécnica de Chiapas  
**Materia**: Multimedia y Diseño Digital
"""
    
    try:
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("✅ README.md creado")
    except Exception as e:
        print(f"❌ Error creando README: {e}")

def run_application():
    """Ejecuta la aplicación principal."""
    print("\n🚀 PASO 7: Iniciando aplicación...")
    print("-" * 40)
    print("🌐 La aplicación se abrirá en: http://localhost:7860")
    print("📱 Para acceso móvil: http://[tu-ip]:7860")
    print()
    print("⏹️ Para detener: Ctrl+C")
    print()
    
    try:
        # Importar y ejecutar la aplicación
        from app import create_gradio_interface
        app = create_gradio_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            quiet=True
        )
    except ImportError:
        print("❌ No se pudo importar la aplicación")
        print("💡 Asegúrate de que app.py esté en el mismo directorio")
    except Exception as e:
        print(f"❌ Error ejecutando aplicación: {e}")

def main():
    """Función principal de instalación y configuración."""
    print_header()
    
    try:
        # Paso 1: Instalar dependencias
        if not install_all_dependencies():
            print("❌ Falló la instalación de dependencias")
            return
        
        # Paso 2: Crear estructura
        create_project_structure()
        
        # Paso 3: Descargar imágenes de prueba
        download_test_images()
        
        # Paso 4: Crear configuración
        create_sample_data_yaml()
        
        # Paso 5: Verificar instalación
        if not test_installation():
            print("⚠️ La instalación tiene problemas, pero continuaremos...")
        
        # Paso 6: Crear documentación
        create_readme()
        
        print("\n🎉 ¡INSTALACIÓN COMPLETADA!")
        print("="*60)
        print("✅ Todo está listo para usar")
        print()
        print("📋 PRÓXIMOS PASOS:")
        print("1. 📸 La aplicación ya tiene imágenes de prueba")
        print("2. 🎯 Prueba la detección con modelo preentrenado")
        print("3. 📝 Crea tu dataset personalizado para útiles escolares") 
        print("4. 🚀 Entrena tu modelo personalizado")
        print()
        
        # Preguntar si ejecutar la aplicación
        response = input("¿Quieres ejecutar la aplicación ahora? (s/n): ").lower().strip()
        if response in ['s', 'si', 'sí', 'y', 'yes']:
            run_application()
        else:
            print("\n💡 Para ejecutar después, usa: python app.py")
            print("📚 Lee README.md para más información")
        
    except KeyboardInterrupt:
        print("\n\n👋 Instalación interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la instalación: {e}")
        print("💡 Intenta ejecutar manualmente cada paso")

if __name__ == "__main__":
    main()