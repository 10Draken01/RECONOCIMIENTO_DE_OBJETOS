"""
Script para solucionar el problema de LabelImg
Crea los archivos necesarios y limpia etiquetas problemáticas
"""

import os
from pathlib import Path
import shutil

def fix_labelimg_files():
    """Solucionar archivos para LabelImg"""
    print("🔧 Solucionando configuración de LabelImg...")
    
    dataset_path = Path("dataset_lapiz")
    
    if not dataset_path.exists():
        print("❌ Dataset no encontrado. Ejecuta primero organize_lapiz_dataset.py")
        return False
    
    # 1. Crear archivo classes.txt en ambas carpetas de labels
    classes_content = "lapiz\n"  # Solo una clase: lapiz
    
    for split in ['train', 'val']:
        labels_dir = dataset_path / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        classes_file = labels_dir / "classes.txt"
        with open(classes_file, 'w', encoding='utf-8') as f:
            f.write(classes_content)
        
        print(f"✅ Creado: {classes_file}")
    
    # 2. Limpiar archivos de etiquetas problemáticos (los que tienen comentarios)
    for split in ['train', 'val']:
        labels_dir = dataset_path / "labels" / split
        
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*.txt"))
            
            for label_file in label_files:
                if label_file.name == "classes.txt":
                    continue  # No tocar classes.txt
                
                try:
                    with open(label_file, 'r') as f:
                        content = f.read()
                    
                    # Si el archivo contiene comentarios, vaciarlo
                    if content.startswith('#') or 'Editar esta etiqueta' in content:
                        with open(label_file, 'w') as f:
                            f.write("")  # Vaciar archivo
                        print(f"🧹 Limpiado: {label_file.name}")
                        
                except Exception as e:
                    print(f"⚠️ Error procesando {label_file}: {e}")
    
    # 3. Crear archivo predefined_classes.txt en la raíz del proyecto
    predefined_classes = dataset_path / "predefined_classes.txt"
    with open(predefined_classes, 'w', encoding='utf-8') as f:
        f.write("lapiz\n")
    
    print(f"✅ Creado: {predefined_classes}")
    
    print("\n✅ ¡Configuración arreglada!")
    return True

def create_labelimg_guide():
    """Crear guía paso a paso para LabelImg"""
    guide_content = """
# 🏷️ GUÍA PASO A PASO PARA LABELIMG

## ✅ PROBLEMA SOLUCIONADO
Ya se crearon los archivos necesarios:
- classes.txt en train/ y val/
- predefined_classes.txt en la raíz

## 📋 PASOS PARA ETIQUETAR:

### 1. 🚀 Abrir LabelImg
```bash
labelImg
```

### 2. ⚙️ Configurar LabelImg
1. **Cambiar a formato YOLO:**
   - Click en "PascalVOC" (esquina inferior izquierda)
   - Cambiar a "YOLO"

2. **Configurar directorios:**
   - Click "Open Dir" → Selecciona: `dataset_lapiz/images/train/`
   - Click "Change Save Dir" → Selecciona: `dataset_lapiz/labels/train/`

3. **Cargar clases predefinidas:**
   - File → Load Predefined Classes
   - Selecciona: `dataset_lapiz/predefined_classes.txt`

### 3. 🎯 Etiquetar imágenes
1. **Por cada imagen:**
   - Dibuja rectángulo alrededor del lápiz (click y arrastra)
   - Selecciona clase "lapiz" 
   - Guarda (Ctrl+S)
   - Siguiente imagen (D)

2. **Consejos de etiquetado:**
   - Rectángulo debe cubrir TODO el lápiz
   - Include punta y goma si están visibles
   - Si hay múltiples lápices, etiqueta TODOS
   - Sé consistente con el tamaño del rectángulo

### 4. 🔄 Repetir para validación
Después de terminar train/:
1. Open Dir → `dataset_lapiz/images/val/`
2. Change Save Dir → `dataset_lapiz/labels/val/`
3. Etiquetar todas las imágenes de validación

### 5. ✅ Verificación
Al final debes tener:
- Cada imagen.jpg con su imagen.txt
- Archivo classes.txt en cada carpeta labels/
- Todas las etiquetas con formato: 0 x y w h

## 🚨 ERRORES COMUNES:
- ❌ Olvidar cambiar a formato YOLO
- ❌ No configurar Change Save Dir
- ❌ Dejar imágenes sin etiquetar
- ❌ Etiquetas fuera del rango 0-1

## 🎉 DESPUÉS DEL ETIQUETADO:
```bash
python train_lapiz_model.py
```
"""
    
    guide_path = Path("GUIA_LABELIMG.md")
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"📖 Guía creada: {guide_path}")

def verify_dataset_structure():
    """Verificar estructura del dataset"""
    print("\n🔍 Verificando estructura del dataset...")
    
    dataset_path = Path("dataset_lapiz")
    
    required_dirs = [
        "images/train",
        "images/val", 
        "labels/train",
        "labels/val"
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"✅ {dir_name}: {file_count} archivos")
        else:
            print(f"❌ Falta: {dir_name}")
            all_good = False
    
    # Verificar classes.txt
    for split in ['train', 'val']:
        classes_file = dataset_path / "labels" / split / "classes.txt"
        if classes_file.exists():
            print(f"✅ classes.txt en {split}")
        else:
            print(f"❌ No existe classes.txt en {split}")
            all_good = False
    
    return all_good

def count_images_and_labels():
    """Contar imágenes y etiquetas"""
    print("\n📊 Contando archivos...")
    
    dataset_path = Path("dataset_lapiz")
    
    for split in ['train', 'val']:
        images_dir = dataset_path / "images" / split
        labels_dir = dataset_path / "labels" / split
        
        if images_dir.exists() and labels_dir.exists():
            # Contar imágenes
            image_files = (list(images_dir.glob("*.jpg")) + 
                          list(images_dir.glob("*.jpeg")) + 
                          list(images_dir.glob("*.png")))
            
            # Contar etiquetas (excluyendo classes.txt)
            label_files = [f for f in labels_dir.glob("*.txt") 
                          if f.name != "classes.txt"]
            
            print(f"📂 {split.upper()}:")
            print(f"   • Imágenes: {len(image_files)}")
            print(f"   • Etiquetas: {len(label_files)}")
            
            # Verificar que cada imagen tenga potencial etiqueta
            missing_labels = []
            for img_file in image_files:
                expected_label = labels_dir / f"{img_file.stem}.txt"
                if not expected_label.exists():
                    missing_labels.append(img_file.name)
            
            if missing_labels:
                print(f"   ⚠️ Sin etiqueta: {len(missing_labels)} imágenes")
            else:
                print(f"   ✅ Todas las imágenes tienen archivo de etiqueta")

def main():
    """Función principal"""
    print("🔧 SOLUCIONADOR DE PROBLEMAS LABELIMG")
    print("="*50)
    
    # Verificar que existe el dataset
    if not Path("dataset_lapiz").exists():
        print("❌ Dataset no encontrado")
        print("💡 Ejecuta primero: python organize_lapiz_dataset.py")
        return
    
    # Solucionar archivos de LabelImg
    if fix_labelimg_files():
        print("\n✅ ¡Problema solucionado!")
    else:
        print("\n❌ No se pudo solucionar el problema")
        return
    
    # Crear guía
    create_labelimg_guide()
    
    # Verificar estructura
    if verify_dataset_structure():
        print("\n✅ Estructura correcta")
    else:
        print("\n❌ Hay problemas en la estructura")
    
    # Contar archivos
    count_images_and_labels()
    
    print("\n" + "="*50)
    print("🎉 TODO LISTO PARA LABELIMG")
    print("="*50)
    print("\n📋 PRÓXIMOS PASOS:")
    print("1. 🚀 Ejecuta: labelImg")
    print("2. ⚙️ Configura según GUIA_LABELIMG.md")
    print("3. 🏷️ Etiqueta TODAS las imágenes")
    print("4. 🚀 Ejecuta: python train_lapiz_model.py")
    
    print("\n💡 IMPORTANTE:")
    print("• Cambia a formato YOLO en LabelImg")
    print("• Usa Change Save Dir para especificar donde guardar")
    print("• Etiqueta TODOS los lápices en cada imagen")
    print("• Sé consistente con el tamaño de los rectángulos")

if __name__ == "__main__":
    main()