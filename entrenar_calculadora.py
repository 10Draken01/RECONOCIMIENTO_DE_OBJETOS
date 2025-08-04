#!/usr/bin/env python3
"""
Entrenamiento COHERENTE: Kit de Reciclaje
Universidad Politécnica de Chiapas

LÓGICA COHERENTE:
🟢 PREENTRENADO: botellas individuales + tijeras individuales
🔵 PERSONALIZADO: kit_reciclaje (botella + tijeras juntas)

CONCEPTO: Reciclaje Creativo
- Botella = material reciclable
- Tijeras = herramienta para manualidades  
- Kit = ambos juntos para hacer manualidades reciclando
"""

import os
import yaml
from ultralytics import YOLO
import shutil
from datetime import datetime

class EntrenadorKitReciclaje:
    """Entrena modelo YOLO para detectar kits de reciclaje."""
    
    def __init__(self):
        self.project_name = "kit_reciclaje_detector"
        self.setup_directories()
    
    def setup_directories(self):
        """Crea estructura de carpetas para kit de reciclaje."""
        dirs = [
            "dataset_kit_reciclaje/images/train",
            "dataset_kit_reciclaje/images/val", 
            "dataset_kit_reciclaje/labels/train",
            "dataset_kit_reciclaje/labels/val",
            "models",
            "training_results"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 {dir_path}/")
    
    def create_config_yaml(self):
        """Crea archivo de configuración YAML."""
        config = {
            'path': os.path.abspath('dataset_kit_reciclaje'),
            'train': 'images/train',
            'val': 'images/val', 
            'nc': 1,  # Solo 1 clase: kit_reciclaje
            'names': {0: 'kit_reciclaje'}
        }
        
        yaml_path = "dataset_kit_reciclaje/data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"✅ Configuración creada: {yaml_path}")
        return yaml_path
    
    def validate_dataset(self):
        """Valida el dataset de kits de reciclaje."""
        train_imgs = len([f for f in os.listdir("dataset_kit_reciclaje/images/train") 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_imgs = len([f for f in os.listdir("dataset_kit_reciclaje/images/val")
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        train_labels = len([f for f in os.listdir("dataset_kit_reciclaje/labels/train")
                           if f.endswith('.txt')])
        val_labels = len([f for f in os.listdir("dataset_kit_reciclaje/labels/val")
                         if f.endswith('.txt')])
        
        print(f"📊 Dataset Kit de Reciclaje:")
        print(f"   Imágenes entrenamiento: {train_imgs}")
        print(f"   Etiquetas entrenamiento: {train_labels}")
        print(f"   Imágenes validación: {val_imgs}")
        print(f"   Etiquetas validación: {val_labels}")
        
        if train_imgs < 40:
            print("⚠️ NECESITAS MÁS IMÁGENES")
            print("   Mínimo: 40 imágenes con botella Y tijeras")
            print("   Recomendado: 60+ imágenes")
            return False
        
        if train_imgs != train_labels:
            print("⚠️ Número de imágenes y etiquetas no coincide")
            return False
        
        print("✅ Dataset válido para entrenamiento")
        return True
    
    def train_model(self, epochs=100):
        """Entrena el modelo para detectar kits de reciclaje."""
        if not self.validate_dataset():
            print("❌ Dataset inválido. Corrige los problemas primero.")
            return False
        
        print("♻️ Iniciando entrenamiento de kits de reciclaje...")
        
        # Cargar modelo base
        model = YOLO('yolov8n.pt')
        
        # Configuración
        config_yaml = "dataset_kit_reciclaje/data.yaml"
        
        try:
            # Entrenar
            results = model.train(
                data=config_yaml,
                epochs=epochs,
                imgsz=640,
                batch=16,
                lr0=0.01,
                patience=50,
                project="training_results",
                name=self.project_name,
                exist_ok=True,
                verbose=True
            )
            
            # Copiar mejor modelo
            best_model = f"training_results/{self.project_name}/weights/best.pt"
            final_model = "models/kit_reciclaje_model.pt"
            
            if os.path.exists(best_model):
                shutil.copy2(best_model, final_model)
                print(f"✅ Modelo kit de reciclaje guardado: {final_model}")
                return True
            else:
                print("❌ No se encontró el modelo entrenado")
                return False
                
        except Exception as e:
            print(f"❌ Error durante entrenamiento: {e}")
            return False
    
    def create_instructions(self):
        """Crea guía específica para kits de reciclaje."""
        instructions = """
# ♻️ GUÍA: Entrenar Detector de Kits de Reciclaje

## 🎯 CONCEPTO DEL MODELO PERSONALIZADO

### ¿Qué es un "Kit de Reciclaje"?
Un **kit de reciclaje** es cuando tienes:
- 🍼 **Una botella** (material reciclable)
- ✂️ **Unas tijeras** (herramienta para cortar)
- 🤝 **Ambos objetos juntos** en la misma imagen

**Propósito**: Detectar cuando alguien está listo para hacer manualidades reciclando plástico.

## 📸 PASO 1: Recolectar Imágenes (3-4 horas)

### ¿Qué fotografiar?
**60 imágenes de "kits completos"** con botella Y tijeras visibles:

#### 🏠 Escenarios en casa:
- Mesa de cocina con botella de agua + tijeras de cocina
- Escritorio con botella de refresco + tijeras de oficina
- Mesa de manualidades con ambos objetos preparados

#### 🏫 Escenarios escolares:
- Proyecto escolar de reciclaje
- Taller de manualidades con materiales listos
- Mesa de trabajo con "kit" preparado

#### 🎨 Escenarios de manualidades:
- Antes de cortar una botella para hacer maceta
- Preparación para hacer organizador de botellas
- Kit listo para proyecto de reciclaje creativo

### 💡 Composiciones recomendadas:
1. **Botella horizontal + tijeras al lado** (más fácil)
2. **Botella vertical + tijeras delante** (vista natural)
3. **Ambos en estuche o caja** (kit organizado)
4. **Sobre mesa con otros materiales** (contexto real)
5. **En manos preparándose** para usar (acción)

### ⚠️ IMPORTANTE para el éxito:
- **Ambos objetos SIEMPRE visibles** en la misma imagen
- **Distancia máxima**: 50cm entre botella y tijeras
- **Una sola etiqueta**: "kit_reciclaje" para AMBOS objetos juntos
- **Contexto coherente**: Escenario de manualidades/reciclaje

## 🏷️ PASO 2: Anotar Correctamente (2-3 horas)

### Estrategia de anotación CLAVE:

#### ✅ CORRECTO:
```
Una imagen con botella + tijeras = UNA etiqueta "kit_reciclaje"
La caja debe incluir AMBOS objetos
```

#### ❌ INCORRECTO:
```
Una caja para botella + otra caja para tijeras
Anotar solo uno de los objetos
Cajas separadas
```

### Proceso en LabelImg:
1. Abrir imagen con botella Y tijeras
2. Crear UNA caja grande que incluya AMBOS objetos
3. Etiquetar como "kit_reciclaje"
4. NO crear cajas separadas para cada objeto
5. Guardar y continuar

### Ejemplo visual:
```
[Imagen: botella en mesa + tijeras al lado]
Anotación: [----caja grande----] = "kit_reciclaje"
           [botella] [tijeras]
```

## 📁 PASO 3: Organizar Dataset

### Estructura requerida:
```
dataset_kit_reciclaje/
├── data.yaml
├── images/
│   ├── train/          # 48 imágenes (80%)
│   └── val/            # 12 imágenes (20%)
└── labels/
    ├── train/          # 48 etiquetas (.txt)
    └── val/            # 12 etiquetas (.txt)
```

### División recomendada:
- **Entrenamiento (80%)**: Imágenes más variadas y claras
- **Validación (20%)**: Imágenes diferentes para prueba

## 🚀 PASO 4: Entrenar Modelo

```bash
python entrenar_kit_reciclaje.py
```

### Configuración óptima:
- **Epochs**: 100-150
- **Batch size**: 16
- **Learning rate**: 0.01
- **Tiempo**: 45-90 minutos

### Métricas objetivo:
- **mAP50 > 0.6**: Aceptable para concepto nuevo
- **mAP50 > 0.75**: Excelente rendimiento
- **Precisión > 0.7**: Pocas falsas detecciones
- **Recall > 0.6**: Detecta la mayoría de kits reales

## ✅ PASO 5: Probar en Aplicación

1. Ejecutar: `python app.py`
2. Subir imagen con botella + tijeras juntas
3. ¡Ver caja azul con "♻️ KIT-RECICLAJE"!

## 🎯 Ventajas de Este Concepto

### ✅ **Coherencia lógica**:
- Objetos individuales → Concepto integrado
- Materiales + herramientas → Kit completo
- Reciclaje individual → Reciclaje organizado

### ✅ **Practicidad**:
- Fácil conseguir objetos (botella + tijeras)
- Escenarios naturales y realistas
- Aplicación práctica (detección de preparación)

### ✅ **Evaluación académica**:
- Demuestra comprensión de machine learning
- Concepto original y creativo
- Integración lógica de componentes

## 🆘 Solución de Problemas

### "El modelo no detecta kits"
- Verificar que las cajas incluyan AMBOS objetos
- Aumentar variedad de escenarios
- Revisar que la distancia entre objetos sea <50cm

### "Detecta kits donde no los hay"
- Añadir imágenes negativas (solo botella O solo tijeras)
- Mejorar precisión de las anotaciones
- Entrenar más epochs

### "mAP muy bajo"
- Verificar consistencia en anotaciones
- Asegurar que todas las imágenes tengan ambos objetos
- Mejorar calidad/claridad de las fotos

## 💡 Consejos Finales

### Para máximo éxito:
1. **Consistencia**: Siempre anotar de la misma manera
2. **Variedad**: Diferentes botellas, diferentes tijeras
3. **Contexto**: Escenarios realistas de reciclaje
4. **Calidad**: Fotos claras con buena iluminación

### Para la presentación:
- Explicar la lógica: objetos individuales → kit integrado
- Mostrar detección funcionando en vivo
- Destacar la aplicación práctica del concepto
"""
        
        with open("GUIA_KIT_RECICLAJE.md", "w", encoding="utf-8") as f:
            f.write(instructions)
        
        print("📋 Guía completa creada: GUIA_KIT_RECICLAJE.md")

def main():
    """Función principal."""
    print("♻️ ENTRENADOR DE KITS DE RECICLAJE")
    print("="*60)
    print("Universidad Politécnica de Chiapas")
    print("Proyecto: Reciclaje Creativo Coherente")
    print()
    
    trainer = EntrenadorKitReciclaje()
    
    # Crear configuración
    trainer.create_config_yaml()
    
    # Crear guía
    trainer.create_instructions()
    
    print("\n🎯 CONCEPTO DEL PROYECTO:")
    print("   🟢 INDIVIDUALES: Botellas + Tijeras")
    print("   🔵 INTEGRADO: Kit de Reciclaje")
    print("   💡 LÓGICA: Material + Herramienta = Kit Completo")
    print()
    print("📋 PRÓXIMOS PASOS:")
    print("1. 📖 Lee GUIA_KIT_RECICLAJE.md")  
    print("2. 📸 Recolecta 60+ fotos de botella+tijeras juntas")
    print("3. 🏷️ Anota como UNA clase: 'kit_reciclaje'")
    print("4. 📁 Organiza en dataset_kit_reciclaje/")
    print("5. 🚀 Ejecuta este script nuevamente")
    
    # Preguntar si quiere entrenar
    if input("\n¿Tienes el dataset de kits listo? (s/n): ").lower() == 's':
        print("♻️ Iniciando entrenamiento...")
        success = trainer.train_model()
        
        if success:
            print("\n🎉 ¡ENTRENAMIENTO EXITOSO!")
            print("✅ Modelo guardado: models/kit_reciclaje_model.pt")
            print("🎯 Ahora ejecuta: python app.py")
            print("♻️ Verás cajas azules detectando kits de reciclaje")
            print("\n🏆 PROYECTO 100% COMPLETO:")
            print("   🟢 Botellas individuales")
            print("   🟢 Tijeras individuales") 
            print("   🔵 Kits de reciclaje")
        else:
            print("\n❌ Entrenamiento falló. Revisa el dataset.")
    else:
        print("\n📖 Lee GUIA_KIT_RECICLAJE.md y regresa cuando tengas el dataset")
        print("💡 Recuerda: UNA caja para botella+tijeras = 'kit_reciclaje'")

if __name__ == "__main__":
    main()