"""
Script para entrenar modelo personalizado de señales de alto mexicanas
Descripción: Entrena un modelo YOLOv8 personalizado para detectar señales de alto 
"""
import os
import yaml
from ultralytics import YOLO
import torch
from pathlib import Path
import shutil
from datetime import datetime

class CustomModelTrainer:
    """
    Clase para entrenar el modelo personalizado de señales de alto
    """
    
    def __init__(self, dataset_path: str = "dataset"):
        """
        Inicializa el entrenador
        
        Args:
            dataset_path: Ruta al dataset organizado
        """
        self.dataset_path = Path(dataset_path)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Configuración de entrenamiento
        self.config = {
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'patience': 20,  # Early stopping
            'save_period': 10,  # Guardar checkpoint cada 10 epochs
        }
        
    def create_dataset_yaml(self) -> str:
        """
        Crea el archivo YAML de configuración del dataset
        
        Returns:
            str: Ruta al archivo YAML creado
        """
        yaml_config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',  # Opcional
            'nc': 1,  # Número de clases
            'names': ['señal_alto']  # Nombres de las clases
        }
        
        yaml_path = self.dataset_path / "dataset.yaml"
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"📝 Archivo de configuración creado: {yaml_path}")
        return str(yaml_path)
    
    def validate_dataset_structure(self) -> bool:
        """
        Valida que el dataset tenga la estructura correcta
        
        Returns:
            bool: True si la estructura es válida
        """
        required_dirs = [
            self.dataset_path / "images" / "train",
            self.dataset_path / "images" / "val",
            self.dataset_path / "labels" / "train",
            self.dataset_path / "labels" / "val"
        ]
        
        print("🔍 Validando estructura del dataset...")
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                print(f"❌ Directorio faltante: {dir_path}")
                return False
            else:
                file_count = len(list(dir_path.glob("*")))
                print(f"✅ {dir_path}: {file_count} archivos")
        
        # Validar que hay suficientes imágenes de entrenamiento
        train_images = list((self.dataset_path / "images" / "train").glob("*"))
        if len(train_images) < 50:
            print(f"⚠️  Advertencia: Solo {len(train_images)} imágenes de entrenamiento. Se recomiendan al menos 50.")
        
        return True
    
    def train_model(self, base_model: str = 'yolov8n.pt') -> str:
        """
        Entrena el modelo personalizado
        
        Args:
            base_model: Modelo base para transfer learning
            
        Returns:
            str: Ruta al modelo entrenado
        """
        print(f"🚀 Iniciando entrenamiento del modelo personalizado...")
        print(f"📊 Configuración: {self.config}")
        
        # Crear archivo YAML del dataset
        yaml_path = self.create_dataset_yaml()
        
        # Cargar modelo base
        model = YOLO(base_model)
        
        # Configurar entrenamiento
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"senales_alto_{timestamp}"
        
        # Entrenar modelo
        results = model.train(
            data=yaml_path,
            epochs=self.config['epochs'],
            batch=self.config['batch_size'],
            imgsz=self.config['img_size'],
            device=self.config['device'],
            project=str(self.models_dir),
            name=project_name,
            patience=self.config['patience'],
            save_period=self.config['save_period'],
            verbose=True,
            val=True,
            plots=True,  # Generar gráficas de entrenamiento
            save_json=True,  # Guardar métricas en JSON
        )
        
        # Ruta del mejor modelo
        best_model_path = self.models_dir / project_name / "weights" / "best.pt"
        
        # Copiar el mejor modelo a la ubicación esperada por la aplicación
        final_model_path = self.models_dir / "senales_alto.pt"
        if best_model_path.exists():
            shutil.copy2(best_model_path, final_model_path)
            print(f"✅ Modelo final guardado en: {final_model_path}")
        
        # Mostrar métricas finales
        self.show_training_summary(results, project_name)
        
        return str(final_model_path)
    
    def show_training_summary(self, results, project_name: str):
        """Muestra un resumen del entrenamiento"""
        print("\n" + "="*60)
        print("🎯 RESUMEN DEL ENTRENAMIENTO")
        print("="*60)
        
        try:
            # Obtener métricas del último epoch
            metrics = results.results_dict
            
            print(f"📈 Métricas finales:")
            print(f"   • Precisión (P): {metrics.get('metrics/precision(B)', 0):.3f}")
            print(f"   • Recall (R): {metrics.get('metrics/recall(B)', 0):.3f}")
            print(f"   • mAP@0.5: {metrics.get('metrics/mAP50(B)', 0):.3f}")
            print(f"   • mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.3f}")
            
            print(f"\n📁 Archivos generados en: {self.models_dir / project_name}")
            print("   • best.pt - Mejor modelo")
            print("   • last.pt - Último modelo")
            print("   • results.csv - Métricas por epoch")
            print("   • confusion_matrix.png - Matriz de confusión")
            print("   • results.png - Gráficas de entrenamiento")
            
        except Exception as e:
            print(f"No se pudieron mostrar las métricas: {e}")
        
        print("="*60)
    
    def test_model(self, model_path: str, test_images_dir: str = None):
        """
        Prueba el modelo entrenado con imágenes de prueba
        
        Args:
            model_path: Ruta al modelo entrenado
            test_images_dir: Directorio con imágenes de prueba
        """
        if not test_images_dir:
            test_images_dir = self.dataset_path / "images" / "test"
        
        if not Path(test_images_dir).exists():
            print(f"⚠️  No se encontró directorio de pruebas: {test_images_dir}")
            return
        
        print(f"🧪 Probando modelo con imágenes en: {test_images_dir}")
        
        # Cargar modelo
        model = YOLO(model_path)
        
        # Obtener imágenes de prueba
        test_images = list(Path(test_images_dir).glob("*.jpg")) + \
                     list(Path(test_images_dir).glob("*.png"))
        
        if not test_images:
            print("❌ No se encontraron imágenes de prueba")
            return
        
        print(f"📸 Procesando {len(test_images)} imágenes de prueba...")
        
        # Procesar cada imagen
        for img_path in test_images[:5]:  # Procesar solo las primeras 5
            results = model(str(img_path))
            
            # Mostrar resultados
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    print(f"✅ {img_path.name}: {len(boxes)} señales detectadas")
                    for box in boxes:
                        conf = box.conf[0].cpu().numpy()
                        print(f"   • Confianza: {conf:.3f}")
                else:
                    print(f"❌ {img_path.name}: No se detectaron señales")

def create_dataset_structure(base_path: str = "dataset"):
    """
    Crea la estructura básica del dataset
    
    Args:
        base_path: Ruta base donde crear el dataset
    """
    dataset_path = Path(base_path)
    
    # Crear directorios
    dirs_to_create = [
        dataset_path / "images" / "train",
        dataset_path / "images" / "val",
        dataset_path / "images" / "test",
        dataset_path / "labels" / "train",
        dataset_path / "labels" / "val",
        dataset_path / "labels" / "test"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Creado: {dir_path}")
    
    # Crear archivo README con instrucciones
    readme_content = """
# Dataset de Señales de Alto Mexicanas

## Estructura del Dataset:
```
dataset/
├── images/
│   ├── train/          # Imágenes de entrenamiento (70%)
│   ├── val/            # Imágenes de validación (20%)
│   └── test/           # Imágenes de prueba (10%)
└── labels/
    ├── train/          # Etiquetas de entrenamiento (.txt)
    ├── val/            # Etiquetas de validación (.txt)
    └── test/           # Etiquetas de prueba (.txt)
```

## Formato de Etiquetas (YOLO):
Cada archivo .txt debe contener una línea por objeto:
```
class_id center_x center_y width height
```

Para señales de alto: `0 0.5 0.5 0.3 0.4`

## Recomendaciones:
- Mínimo 50 imágenes de entrenamiento
- Imágenes variadas: diferentes ángulos, iluminación, distancias
- Etiquetas precisas y consistentes
- Distribución: 70% train, 20% val, 10% test
"""
    
    with open(dataset_path / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"\n✅ Estructura del dataset creada en: {dataset_path}")
    print("📖 Lee el archivo README.md para instrucciones detalladas")

def main():
    """Función principal para entrenar el modelo"""
    print("🎯 Entrenador de Modelo Personalizado - Señales de Alto Mexicanas")
    print("="*70)
    
    # Crear estructura del dataset si no existe
    if not Path("dataset").exists():
        print("📁 Creando estructura del dataset...")
        create_dataset_structure()
        print("\n⚠️  IMPORTANTE: Agrega tus imágenes y etiquetas antes de continuar.")
        print("📖 Consulta el archivo dataset/README.md para más información.")
        return
    
    # Inicializar entrenador
    trainer = CustomModelTrainer()
    
    # Validar dataset
    if not trainer.validate_dataset_structure():
        print("❌ La estructura del dataset no es válida. Revisa los directorios.")
        return
    
    # Entrenar modelo
    try:
        model_path = trainer.train_model()
        print(f"\n🎉 ¡Entrenamiento completado exitosamente!")
        print(f"📦 Modelo guardado en: {model_path}")
        
        # Probar modelo si hay imágenes de prueba
        trainer.test_model(model_path)
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return
    
    print("\n🚀 El modelo está listo para usar en la aplicación principal!")

if __name__ == "__main__":
    main()