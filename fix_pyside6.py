"""
Script para verificar y corregir problemas de PySide6
Ejecutar este script primero para verificar instalación
"""

import sys

def check_pyside6():
    """Verificar instalación de PySide6"""
    print("🔍 Verificando PySide6...")
    
    try:
        import PySide6
        print(f"✅ PySide6 versión: {PySide6.__version__}")
        
        # Verificar imports específicos
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Signal, QThread
        from PySide6.QtGui import QPixmap
        
        print("✅ Todos los imports necesarios funcionan")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False

def check_ultralytics():
    """Verificar YOLO"""
    print("🔍 Verificando Ultralytics...")
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics funcionando")
        
        # Intentar cargar modelo básico
        model = YOLO('yolov8n.pt')
        print("✅ Modelo YOLO cargado correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error con YOLO: {e}")
        return False

def check_opencv():
    """Verificar OpenCV"""
    print("🔍 Verificando OpenCV...")
    
    try:
        import cv2
        print(f"✅ OpenCV versión: {cv2.__version__}")
        return True
        
    except ImportError as e:
        print(f"❌ Error con OpenCV: {e}")
        return False

def install_missing():
    """Instalar dependencias faltantes"""
    print("📦 Instalando dependencias faltantes...")
    
    import subprocess
    
    packages = [
        "PySide6>=6.5.0",
        "ultralytics>=8.0.0", 
        "opencv-python>=4.5.0",
        "torch>=1.9.0",
        "numpy>=1.21.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Instalado: {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Error instalando: {package}")

def main():
    """Función principal de verificación"""
    print("🛠️ VERIFICADOR DE DEPENDENCIAS")
    print("="*40)
    
    all_good = True
    
    # Verificar cada componente
    if not check_pyside6():
        all_good = False
    
    if not check_ultralytics():
        all_good = False
        
    if not check_opencv():
        all_good = False
    
    if all_good:
        print("\n🎉 ¡Todo funciona correctamente!")
        print("🚀 Puedes ejecutar: python yolo_detector_app.py")
    else:
        print("\n⚠️ Hay problemas con las dependencias")
        response = input("¿Quieres intentar instalar automáticamente? (s/n): ")
        
        if response.lower() in ['s', 'si', 'y', 'yes']:
            install_missing()
            print("\n🔄 Volviendo a verificar...")
            main()  # Verificar de nuevo
        else:
            print("💡 Instala manualmente las dependencias faltantes")

if __name__ == "__main__":
    main()