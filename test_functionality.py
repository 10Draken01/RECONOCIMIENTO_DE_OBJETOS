#!/usr/bin/env python3
"""
Script de prueba funcional completa
Universidad Politécnica de Chiapas

Este script verifica que TODA la funcionalidad funcione correctamente:
- Importaciones
- Modelos YOLO  
- Procesamiento de imágenes
- Procesamiento de videos
- Interfaz Gradio
- Exportación de datos
"""

import sys
import os
import numpy as np
import tempfile
import time

def test_imports():
    """Verifica que todas las importaciones funcionen."""
    print("🔍 TEST 1: Verificando importaciones...")
    print("-" * 40)
    
    imports_to_test = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("gradio", "Gradio"),
        ("ultralytics", "Ultralytics YOLO"),
        ("torch", "PyTorch"),
        ("matplotlib.pyplot", "Matplotlib"),
        ("json", "JSON"),
        ("datetime", "DateTime")
    ]
    
    failed_imports = []
    
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            failed_imports.append(display_name)
    
    if failed_imports:
        print(f"\n⚠️ Faltan {len(failed_imports)} módulos: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ Todas las importaciones funcionan correctamente")
        return True

def test_yolo_model():
    """Verifica que YOLO funcione correctamente."""
    print("\n🤖 TEST 2: Verificando modelo YOLO...")
    print("-" * 40)
    
    try:
        from ultralytics import YOLO
        
        # Cargar modelo
        print("📥 Cargando modelo YOLOv8...")
        model = YOLO('yolov8n.pt')
        print("✅ Modelo cargado correctamente")
        
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Hacer predicción
        print("🔍 Realizando detección de prueba...")
        results = model(test_image, verbose=False)
        print("✅ Detección funcionando")
        
        # Verificar estructura de resultados
        if results and len(results) > 0:
            result = results[0]
            print(f"✅ Estructura de resultados válida")
            print(f"   📊 Clases disponibles: {len(model.names)} clases")
            return True
        else:
            print("⚠️ No hay resultados, pero el modelo funciona")
            return True
            
    except Exception as e:
        print(f"❌ Error con YOLO: {e}")
        return False

def test_image_processing():
    """Verifica el procesamiento de imágenes."""
    print("\n📸 TEST 3: Verificando procesamiento de imágenes...")
    print("-" * 40)
    
    try:
        # Importar el detector
        sys.path.append('.')
        from app import YOLOObjectDetector
        
        # Crear detector
        print("🔧 Creando detector...")
        detector = YOLOObjectDetector()
        print("✅ Detector creado")
        
        # Crear imagen de prueba simulando útiles escolares
        test_image = create_test_image()
        print("✅ Imagen de prueba creada")
        
        # Procesar imagen
        print("🔍 Procesando imagen...")
        result_image, stats, json_results = detector.process_single_image(test_image, 0.5)
        
        if result_image is not None:
            print("✅ Procesamiento de imagen exitoso")
            print(f"📊 Estadísticas generadas: {len(stats)} caracteres")
            return True
        else:
            print("❌ Error en procesamiento de imagen")
            return False
            
    except Exception as e:
        print(f"❌ Error en procesamiento de imagen: {e}")
        return False

def test_video_processing():
    """Verifica el procesamiento de videos."""
    print("\n🎬 TEST 4: Verificando procesamiento de videos...")
    print("-" * 40)
    
    try:
        import cv2
        
        # Crear video de prueba temporal
        print("🎬 Creando video de prueba...")
        test_video_path = create_test_video()
        
        if not os.path.exists(test_video_path):
            print("❌ No se pudo crear video de prueba")
            return False
        
        print("✅ Video de prueba creado")
        
        # Importar detector
        from app import YOLOObjectDetector
        detector = YOLOObjectDetector()
        
        # Procesar video (solo primeros 30 frames para prueba rápida)
        print("🔍 Procesando video...")
        result_video, stats, json_results = detector.process_video(test_video_path, 0.5, 30)
        
        # Limpiar archivo temporal
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
        
        if result_video and os.path.exists(result_video):
            print("✅ Procesamiento de video exitoso")
            # Limpiar video de resultado
            os.remove(result_video)
            return True
        else:
            print("❌ Error en procesamiento de video")
            return False
            
    except Exception as e:
        print(f"❌ Error en procesamiento de video: {e}")
        return False

def test_gradio_interface():
    """Verifica que la interfaz Gradio se pueda crear."""
    print("\n🌐 TEST 5: Verificando interfaz Gradio...")
    print("-" * 40)
    
    try:
        from app import create_gradio_interface
        
        print("🔧 Creando interfaz Gradio...")
        interface = create_gradio_interface()
        
        if interface:
            print("✅ Interfaz Gradio creada correctamente")
            return True
        else:
            print("❌ Error creando interfaz Gradio")
            return False
            
    except Exception as e:
        print(f"❌ Error con interfaz Gradio: {e}")
        return False

def test_csv_export():
    """Verifica la funcionalidad de exportación CSV."""
    print("\n📊 TEST 6: Verificando exportación CSV...")
    print("-" * 40)
    
    try:
        from app import YOLOObjectDetector
        
        # Crear detector y simular datos
        detector = YOLOObjectDetector()
        detector.detection_results = {
            'pretrained': {'book': 2, 'person': 1},
            'custom': {'lapiz': 3, 'cuaderno': 1},
            'timestamp': '2024-01-01T10:00:00',
            'total_frames': 1,
            'processing_time': 1.5
        }
        
        # Exportar CSV
        print("📤 Exportando CSV...")
        csv_path = detector.export_results_csv()
        
        if csv_path and os.path.exists(csv_path):
            print(f"✅ CSV exportado: {csv_path}")
            # Limpiar archivo
            os.remove(csv_path)
            return True
        else:
            print("❌ Error exportando CSV")
            return False
            
    except Exception as e:
        print(f"❌ Error en exportación CSV: {e}")
        return False

def create_test_image():
    """Crea una imagen de prueba simulando útiles escolares."""
    import cv2
    
    # Crear imagen base
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Simular cuaderno (rectángulo azul)
    cv2.rectangle(image, (100, 200), (300, 350), (255, 100, 100), -1)
    
    # Simular lápiz (línea amarilla)
    cv2.line(image, (350, 250), (500, 230), (0, 255, 255), 10)
    
    # Simular persona (círculo)
    cv2.circle(image, (500, 100), 50, (200, 180, 160), -1)
    
    return image

def create_test_video():
    """Crea un video de prueba temporal."""
    import cv2
    
    # Archivo temporal
    temp_video = tempfile.mktemp(suffix='.mp4')
    
    # Configuración
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, 10, (640, 480))
    
    # Crear 30 frames
    for i in range(30):
        frame = create_test_image()
        # Añadir variación por frame
        cv2.putText(frame, f"Frame {i+1}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        out.write(frame)
    
    out.release()
    return temp_video

def run_all_tests():
    """Ejecuta todos los tests y genera reporte."""
    print("🧪 EJECUTANDO TESTS FUNCIONALES COMPLETOS")
    print("="*60)
    print("Universidad Politécnica de Chiapas")
    print("Proyecto: Detector de Útiles Escolares")
    print("="*60)
    print()
    
    tests = [
        ("Importaciones", test_imports),
        ("Modelo YOLO", test_yolo_model),
        ("Procesamiento de Imágenes", test_image_processing),
        ("Procesamiento de Videos", test_video_processing),
        ("Interfaz Gradio", test_gradio_interface),
        ("Exportación CSV", test_csv_export)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_function in tests:
        try:
            result = test_function()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ ERROR CRÍTICO en {test_name}: {e}")
            results.append((test_name, False))
    
    total_time = time.time() - start_time
    
    # Generar reporte final
    print("\n" + "="*60)
    print("📊 REPORTE FINAL DE TESTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    success_rate = (passed / total) * 100
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{status:10} {test_name}")
    
    print("-" * 60)
    print(f"📊 RESULTADOS: {passed}/{total} tests pasaron ({success_rate:.1f}%)")
    print(f"⏱️ TIEMPO TOTAL: {total_time:.2f} segundos")
    
    if success_rate >= 80:
        print("\n🎉 ¡SISTEMA FUNCIONAL!")
        print("✅ La aplicación está lista para usar")
        print("🚀 Ejecuta: python app.py")
    elif success_rate >= 60:
        print("\n⚠️ SISTEMA PARCIALMENTE FUNCIONAL")
        print("💡 Algunas funciones pueden fallar")
        print("🔧 Revisa los tests que fallaron")
    else:
        print("\n❌ SISTEMA CON PROBLEMAS")
        print("🛠️ Necesitas resolver los errores antes de continuar")
        print("📦 Intenta: python install_and_run.py")
    
    return success_rate >= 80

def main():
    """Función principal."""
    try:
        success = run_all_tests()
        
        if success:
            print("\n🎯 ¿QUIERES EJECUTAR LA APLICACIÓN AHORA?")
            response = input("Responde (s/n): ").lower().strip()
            
            if response in ['s', 'si', 'sí', 'y', 'yes']:
                print("\n🚀 Iniciando aplicación...")
                os.system("python app.py")
        
    except KeyboardInterrupt:
        print("\n\n👋 Tests interrumpidos por el usuario")
    except Exception as e:
        print(f"\n❌ Error ejecutando tests: {e}")

if __name__ == "__main__":
    main()