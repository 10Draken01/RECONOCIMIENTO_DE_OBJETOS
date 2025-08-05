"""
Script de entrenamiento para el dataset de Roboflow
Entrena el modelo de lápiz usando el dataset ya etiquetado
"""

import os
import yaml
from ultralytics import YOLO
import torch
from pathlib import Path
import shutil
from datetime import datetime

def verify_roboflow_dataset():
    """Verificar que el dataset de Roboflow esté completo"""
    print("🔍 Verificando dataset de Roboflow...")
    
    dataset_path = Path("dataset_lapiz")
    
    # Verificar estructura
    required_dirs = [
        dataset_path / "train" / "images",
        dataset_path / "train" / "labels",
        dataset_path / "valid" / "images", 
        dataset_path / "valid" / "labels",
        dataset_path / "test" / "images",
        dataset_path / "test" / "labels"
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"❌ Falta directorio: {dir_path}")
            return False
        
        file_count = len(list(dir_path.glob("*")))
        print(f"✅ {dir_path.parent.name}/{dir_path.name}: {file_count} archivos")
    
    # Verificar data.yaml
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        print(f"✅ Archivo data.yaml encontrado")
        
        # Leer contenido
        try:
            with open(yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            print(f"📊 Dataset configuración:")
            print(f"   • Clases: {data_config.get('nc', 'N/A')}")
            print(f"   • Nombres: {data_config.get('names', [])}")
            
        except Exception as e:
            print(f"⚠️ Error leyendo data.yaml: {e}")
    else:
        print("❌ No se encontró data.yaml")
        return False
    
    return True

def validate_sample_labels():
    """Verificar algunas etiquetas para asegurar formato correcto"""
    print("🏷️ Verificando formato de etiquetas...")
    
    dataset_path = Path("dataset_lapiz")
    
    # Verificar algunas etiquetas de train
    train_labels = list((dataset_path / "train" / "labels").glob("*.txt"))
    
    if not train_labels:
        print("❌ No se encontraron etiquetas en train")
        return False
    
    valid_labels = 0
    issues = []
    
    # Verificar primeras 5 etiquetas
    for label_file in train_labels[:5]:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    issues.append(f"{label_file.name}:{line_num} - Formato incorrecto: {line}")
                    continue
                
                try:
                    class_id, x, y, w, h = map(float, parts)
                    
                    # Verificar rangos
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        issues.append(f"{label_file.name}:{line_num} - Coordenadas fuera de rango")
                    else:
                        valid_labels += 1
                        
                except ValueError:
                    issues.append(f"{label_file.name}:{line_num} - Valores no numéricos")
        
        except Exception as e:
            issues.append(f"{label_file.name} - Error leyendo archivo: {e}")
    
    if issues:
        print(f"⚠️ Encontrados {len(issues)} problemas:")
        for issue in issues[:3]:  # Mostrar solo los primeros 3
            print(f"   • {issue}")
        if len(issues) > 3:
            print(f"   ... y {len(issues) - 3} más")
        
        if valid_labels == 0:
            print("❌ No se encontraron etiquetas válidas")
            return False
    
    print(f"✅ Etiquetas válidas encontradas: {valid_labels}")
    return True

def train_lapiz_model():
    """Entrenar modelo de lápiz con dataset de Roboflow"""
    print("🚀 Iniciando entrenamiento del modelo de lápiz...")
    
    dataset_path = Path("dataset_lapiz")
    yaml_path = dataset_path / "data.yaml"
    
    if not yaml_path.exists():
        print(f"❌ No se encontró {yaml_path}")
        return None
    
    # Cargar modelo base YOLOv8
    print("📦 Cargando modelo base YOLOv8n...")
    model = YOLO('yolov8n.pt')
    
    # Configuración de entrenamiento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"lapiz_roboflow_{timestamp}"
    
    print(f"🎯 Configuración de entrenamiento:")
    print(f"   • Proyecto: {project_name}")
    print(f"   • Dataset: {yaml_path}")
    print(f"   • Modelo base: YOLOv8n")
    print(f"   • Epochs: 100")
    print(f"   • Batch size: 16")
    print(f"   • Image size: 640")
    print(f"   • Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"   • Optimizaciones: Activadas")
    
    try:
        print("\n🏃‍♂️ Iniciando entrenamiento...")
        print("⏳ Esto puede tomar 10-30 minutos dependiendo de tu hardware...")
        
        # Entrenar con configuración optimizada
        results = model.train(
            data=str(yaml_path),
            epochs=100,               # Epochs suficientes
            batch=16,                 # Batch size estándar
            imgsz=640,                # Tamaño de imagen estándar
            device='cpu' if not torch.cuda.is_available() else 0,
            project="models",
            name=project_name,
            
            # Parámetros de control
            patience=25,              # Parar si no mejora en 25 epochs
            save_period=10,           # Guardar cada 10 epochs
            
            # Visualización
            verbose=True,
            plots=True,               # Generar gráficas
            
            # Data Augmentation (optimizado para lápices)
            augment=True,
            mixup=0.1,                # Mezcla de imágenes
            copy_paste=0.1,           # Copy-paste augmentation
            
            # Transformaciones geométricas
            degrees=15,               # Rotación ±15 grados
            translate=0.1,            # Translación ±10%
            scale=0.5,                # Escala ±50%
            shear=5.0,                # Shear ±5 grados
            perspective=0.0,          # Sin perspectiva (mejor para objetos simples)
            
            # Flip augmentations
            flipud=0.5,               # Volteo vertical
            fliplr=0.5,               # Volteo horizontal
            
            # Mosaic y otras augmentations
            mosaic=1.0,               # Mosaic augmentation
            
            # Augmentaciones de color (suaves para lápices)
            hsv_h=0.015,              # Variación de matiz
            hsv_s=0.7,                # Variación de saturación  
            hsv_v=0.4,                # Variación de brillo
            
            # Optimización del entrenamiento
            lr0=0.01,                 # Learning rate inicial
            lrf=0.1,                  # Learning rate final
            momentum=0.937,           # Momentum
            weight_decay=0.0005,      # Weight decay
            warmup_epochs=3,          # Epochs de calentamiento
            warmup_momentum=0.8,      # Momentum inicial
            warmup_bias_lr=0.1,       # Learning rate inicial para bias
            
            # Configuraciones adicionales
            box=7.5,                  # Box loss weight
            cls=0.5,                  # Classification loss weight
            dfl=1.5,                  # DFL loss weight
            
            # Validación
            val=True,                 # Validar durante entrenamiento
            save_json=True,           # Guardar resultados en JSON
            
            # Workspace
            exist_ok=True,            # Permitir sobrescribir
            pretrained=True,          # Usar pesos preentrenados
            optimize=False,           # No optimizar para inferencia aún
            keras=False,              # No usar Keras
            resume=False,             # No resumir entrenamiento
            
            # Configuración de workers
            workers=8,                # Número de workers para carga de datos
            
            # Configuración de memoria
            close_mosaic=10           # Desactivar mosaic en últimos N epochs
        )
        
        # Verificar resultados
        best_model_path = Path("models") / project_name / "weights" / "best.pt"
        last_model_path = Path("models") / project_name / "weights" / "last.pt"
        
        if best_model_path.exists():
            # Copiar modelo a ubicación final
            final_model_path = Path("models") / "lapiz_detector.pt"
            shutil.copy2(best_model_path, final_model_path)
            
            print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
            print(f"✅ Mejor modelo guardado en: {final_model_path}")
            
            # Mostrar estadísticas finales
            print(f"\n📊 ESTADÍSTICAS DEL ENTRENAMIENTO:")
            print(f"   • Directorio completo: models/{project_name}")
            print(f"   • Mejor modelo: {best_model_path}")
            print(f"   • Último modelo: {last_model_path}")
            print(f"   • Gráficas de entrenamiento: models/{project_name}/results.png")
            print(f"   • Métricas de validación: models/{project_name}/results.csv")
            
            # Probar el modelo con una imagen de test
            test_model_performance(final_model_path, dataset_path)
            
            return str(final_model_path)
            
        else:
            print("\n❌ No se encontró el modelo entrenado")
            print(f"💡 Revisa la carpeta: models/{project_name}")
            return None
            
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_performance(model_path, dataset_path):
    """Probar el modelo con imágenes de test"""
    print(f"\n🧪 Probando modelo entrenado...")
    
    try:
        # Cargar modelo entrenado
        model = YOLO(str(model_path))
        
        # Buscar imágenes de test
        test_images_dir = dataset_path / "test" / "images"
        test_images = list(test_images_dir.glob("*.jpg"))
        
        if test_images:
            print(f"🖼️ Probando con {len(test_images)} imágenes de test...")
            
            total_detections = 0
            successful_images = 0
            
            for i, test_img in enumerate(test_images[:3]):  # Probar solo 3 imágenes
                try:
                    # Ejecutar predicción
                    results = model(str(test_img), conf=0.5)
                    
                    if results and len(results) > 0:
                        detections = len(results[0].boxes) if results[0].boxes else 0
                        total_detections += detections
                        
                        if detections > 0:
                            successful_images += 1
                            print(f"   ✅ {test_img.name}: {detections} lápices detectados")
                            
                            # Guardar imagen con predicciones
                            annotated = results[0].plot()
                            import cv2
                            output_path = Path("models") / f"test_prediction_{i+1}.jpg"
                            cv2.imwrite(str(output_path), annotated)
                            
                        else:
                            print(f"   ⚠️ {test_img.name}: Sin detecciones")
                    else:
                        print(f"   ❌ {test_img.name}: Error en predicción")
                        
                except Exception as e:
                    print(f"   ❌ {test_img.name}: Error - {e}")
            
            # Resumen de pruebas
            print(f"\n📊 RESUMEN DE PRUEBAS:")
            print(f"   • Imágenes probadas: {min(3, len(test_images))}")
            print(f"   • Imágenes con detecciones: {successful_images}")
            print(f"   • Total de detecciones: {total_detections}")
            print(f"   • Promedio por imagen: {total_detections/min(3, len(test_images)):.1f}")
            
            if successful_images > 0:
                print(f"   ✅ El modelo parece estar funcionando correctamente")
            else:
                print(f"   ⚠️ El modelo no detectó lápices en las imágenes de prueba")
                print(f"   💡 Esto puede ser normal si las imágenes son muy diferentes")
                
        else:
            print("⚠️ No se encontraron imágenes de test")
            
    except Exception as e:
        print(f"❌ Error probando modelo: {e}")

def check_system_requirements():
    """Verificar requisitos del sistema"""
    print("🔍 Verificando requisitos del sistema...")
    
    try:
        import ultralytics
        print(f"✅ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics no encontrado. Instala: pip install ultralytics")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   • GPU: {torch.cuda.get_device_name(0)}")
            print(f"   • Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("❌ PyTorch no encontrado")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV no encontrado")
        return False
    
    return True

def main():
    """Función principal"""
    print("🚀 ENTRENADOR DE MODELO CON DATASET DE ROBOFLOW")
    print("="*60)
    print("🎯 Tu dataset está perfectamente organizado!")
    print("📊 Train: 39 imágenes | Valid: 11 imágenes | Test: 6 imágenes")
    print("="*60)
    
    # Verificar requisitos
    if not check_system_requirements():
        print("\n❌ Faltan requisitos del sistema")
        return
    
    # Verificar dataset
    if not verify_roboflow_dataset():
        print("\n❌ Problemas con el dataset")
        return
    
    # Verificar etiquetas
    if not validate_sample_labels():
        print("\n❌ Problemas con las etiquetas")
        return
    
    # Confirmar entrenamiento
    print(f"\n🎯 ¿Iniciar entrenamiento del modelo de lápiz?")
    print(f"⏳ Tiempo estimado: 15-30 minutos")
    print(f"💾 Espacio necesario: ~500MB")
    
    try:
        choice = input("\n¿Continuar? (s/n): ").lower().strip()
        
        if choice in ['s', 'si', 'sí', 'y', 'yes']:
            print(f"\n🚀 ¡Iniciando entrenamiento!")
            
            model_path = train_lapiz_model()
            
            if model_path:
                print(f"\n🎊 ¡ÉXITO TOTAL!")
                print(f"="*60)
                print(f"✅ Modelo entrenado exitosamente")
                print(f"📦 Ubicación: {model_path}")
                print(f"🎯 Listo para usar en la aplicación")
                
                print(f"\n🚀 PRÓXIMOS PASOS:")
                print(f"1. Ejecuta: python yolo_detector_escritura.py")
                print(f"2. Carga el modelo: {model_path}")
                print(f"3. ¡Prueba detectando lápices y libros!")
                
                print(f"\n💡 RECORDATORIO:")
                print(f"• El modelo detecta LÁPICES (personalizado)")
                print(f"• YOLOv8 detecta LIBROS (preentrenado)")
                print(f"• Cuando están JUNTOS = KIT DE ESCRITURA")
                
            else:
                print(f"\n❌ Entrenamiento falló")
                print(f"💡 Revisa los errores anteriores")
                
        else:
            print(f"👋 Entrenamiento cancelado")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Entrenamiento cancelado por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()