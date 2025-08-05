"""
Script de instalación y configuración automática
Para YOLO Detector con PySide6
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import cv2
import numpy as np
from datetime import datetime

def check_python_version():
    """Verificar versión de Python"""
    print("🐍 Verificando versión de Python...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requerido")
        print(f"   Versión actual: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detectado")
    return True

def install_requirements():
    """Instalar dependencias"""
    print("📦 Instalando dependencias...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def create_directories():
    """Crear estructura de directorios"""
    print("📁 Creando estructura de directorios...")
    
    directories = [
        "models",
        "dataset/images/train",
        "dataset/images/val", 
        "dataset/images/test",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test",
        "output",
        "demo_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")
    
    print("✅ Estructura de directorios creada")

def download_sample_images():
    """Crear imágenes de demostración"""
    print("🎨 Generando imágenes de demostración...")
    
    demo_dir = Path("demo_data")
    
    # Imagen 1: Simulación de personas y autos
    image1 = create_demo_image_1()
    cv2.imwrite(str(demo_dir / "demo_personas_autos.jpg"), image1)
    
    # Imagen 2: Simulación de objetos personalizados
    image2 = create_demo_image_2()
    cv2.imwrite(str(demo_dir / "demo_objetos_reciclaje.jpg"), image2)
    
    # Imagen 3: Imagen mixta
    image3 = create_demo_image_3()
    cv2.imwrite(str(demo_dir / "demo_mixto.jpg"), image3)
    
    print("✅ Imágenes de demostración creadas en demo_data/")

def create_demo_image_1():
    """Crear imagen demo con personas y autos"""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # Fondo gris oscuro
    
    # Simular personas (rectángulos verticales)
    persons = [(100, 200, 140, 400), (200, 180, 230, 420), (600, 220, 630, 450)]
    for x1, y1, x2, y2 in persons:
        cv2.rectangle(img, (x1, y1), (x2, y2), (100, 150, 100), -1)
        cv2.putText(img, "PERSON", (x1-10, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Simular autos (rectángulos horizontales)
    cars = [(300, 350, 500, 420), (150, 480, 350, 550)]
    for x1, y1, x2, y2 in cars:
        cv2.rectangle(img, (x1, y1), (x2, y2), (150, 100, 200), -1)
        # Ruedas
        cv2.circle(img, (x1+30, y2-10), 15, (50, 50, 50), -1)
        cv2.circle(img, (x2-30, y2-10), 15, (50, 50, 50), -1)
        cv2.putText(img, "CAR", (x1+80, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Título
    cv2.putText(img, "DEMO: PERSONAS Y AUTOMOVILES", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return img

def create_demo_image_2():
    """Crear imagen demo con objetos de reciclaje"""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (20, 30, 40)  # Fondo azul oscuro
    
    # Botellas
    bottles = [(100, 200, 130, 350), (200, 180, 225, 380), (300, 220, 325, 390)]
    for x1, y1, x2, y2 in bottles:
        # Cuerpo
        cv2.rectangle(img, (x1, y1+20), (x2, y2), (0, 100, 200), -1)
        # Cuello
        neck_w = (x2-x1)//3
        cv2.rectangle(img, (x1+neck_w, y1), (x2-neck_w, y1+25), (0, 100, 200), -1)
        cv2.putText(img, "BOTTLE", (x1-10, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Tijeras
    scissors_pos = [(450, 250), (550, 350)]
    for x, y in scissors_pos:
        # Mangos
        cv2.rectangle(img, (x, y+50), (x+15, y+120), (150, 150, 150), -1)
        cv2.rectangle(img, (x+20, y+50), (x+35, y+120), (150, 150, 150), -1)
        # Hojas
        points1 = np.array([(x, y), (x+60, y-10), (x+65, y+20), (x+10, y+40)], np.int32)
        points2 = np.array([(x+25, y), (x+85, y-5), (x+90, y+25), (x+35, y+40)], np.int32)
        cv2.fillPoly(img, [points1], (200, 200, 200))
        cv2.fillPoly(img, [points2], (200, 200, 200))
        cv2.putText(img, "SCISSORS", (x-10, y+150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Contenedor de reciclaje
    center = (650, 400)
    cv2.circle(img, center, 50, (0, 150, 0), -1)
    cv2.circle(img, center, 45, (0, 100, 0), 3)
    # Símbolo de reciclaje
    cv2.putText(img, "R", (center[0]-8, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "RECYCLE", (center[0]-30, center[1]+80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Título
    cv2.putText(img, "DEMO: OBJETOS DE RECICLAJE", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return img

def create_demo_image_3():
    """Crear imagen demo mixta"""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (25, 25, 45)  # Fondo púrpura oscuro
    
    # Personas
    cv2.rectangle(img, (50, 200), (80, 400), (100, 150, 100), -1)
    cv2.putText(img, "PERSON", (40, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Auto
    cv2.rectangle(img, (200, 350), (400, 420), (150, 100, 200), -1)
    cv2.circle(img, (230, 410), 15, (50, 50, 50), -1)
    cv2.circle(img, (370, 410), 15, (50, 50, 50), -1)
    cv2.putText(img, "CAR", (280, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Botella
    cv2.rectangle(img, (500, 250), (525, 400), (0, 100, 200), -1)
    cv2.rectangle(img, (505, 240), (520, 260), (0, 100, 200), -1)
    cv2.putText(img, "BOTTLE", (480, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Contenedor
    cv2.circle(img, (650, 300), 40, (0, 150, 0), -1)
    cv2.putText(img, "R", (645, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "RECYCLE", (620, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Título
    cv2.putText(img, "DEMO: DETECCION MIXTA COMPLETA", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return img

def create_readme():
    """Crear archivo README con instrucciones"""
    readme_content = """
# 🚀 YOLO Detector con PySide6

## ⚡ Inicio Rápido

### 1. Ejecutar Aplicación:
```bash
python yolo_detector_app.py
```

### 2. Usar Imágenes Demo:
- Las imágenes están en `demo_data/`
- Usa "📸 CARGAR IMAGEN" para subirlas
- Ajusta umbral de confianza (recomendado: 0.3-0.5)

### 3. Funcionalidades:
- **📸 Imágenes**: Detección instantánea
- **🎬 Videos**: Procesamiento completo 
- **📹 Cámara**: Detección en vivo

### 4. Objetos Detectados:
- 👥 **Personas** (modelo preentrenado)
- 🚗 **Automóviles** (modelo preentrenado)
- 🎯 **Objetos Personalizados** (modelo entrenado)

## 🔧 Solución de Problemas

### Si no detecta objetos:
1. Bajar umbral de confianza a 0.2-0.3
2. Verificar que la imagen tiene buena calidad
3. Asegurar objetos claramente visibles

### Si hay errores de video:
1. Usar formatos estándar (MP4, AVI)
2. Videos cortos para pruebas (< 1 min)
3. Verificar que el archivo no esté corrupto

## 🎯 Modelo Personalizado

Para entrenar tu modelo personalizado:
1. Agregar imágenes a `dataset/`
2. Ejecutar: `python train_recycling_model.py`
3. Reiniciar aplicación

---
*Aplicación desarrollada con YOLO + PySide6*
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ README.md creado")

def test_installation():
    """Probar instalación"""
    print("🧪 Probando instalación...")
    
    try:
        # Probar imports
        import PySide6
        from ultralytics import YOLO
        import torch
        
        print("✅ Todas las dependencias importadas correctamente")
        
        # Probar carga de modelo
        model = YOLO('yolov8n.pt')
        print("✅ Modelo YOLO cargado correctamente")
        
        # Verificar GPU
        if torch.cuda.is_available():
            print(f"🚀 GPU disponible: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Usando CPU (normal)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def main():
    """Función principal de setup"""
    print("🎯 CONFIGURACIÓN AUTOMÁTICA - YOLO DETECTOR")
    print("="*60)
    
    # Verificaciones
    if not check_python_version():
        return False
    
    # Crear directorios
    create_directories()
    
    # Instalar dependencias
    if not install_requirements():
        return False
    
    # Crear contenido demo
    download_sample_images()
    
    # Crear documentación
    create_readme()
    
    # Probar instalación
    if not test_installation():
        return False
    
    print("\n🎉 ¡CONFIGURACIÓN COMPLETADA EXITOSAMENTE!")
    print("="*60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. 🚀 Ejecutar: python yolo_detector_app.py")
    print("2. 📸 Cargar imagen demo desde demo_data/")
    print("3. 🎯 Ajustar umbral de confianza")
    print("4. 🔍 ¡Disfrutar de la detección!")
    print("\n💡 CONSEJO: Inicia con las imágenes demo para probar")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Configuración falló. Revisa los errores anteriores.")
        sys.exit(1)