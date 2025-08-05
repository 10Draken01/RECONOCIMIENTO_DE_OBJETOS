"""
Script para organizar las imágenes de lápiz en estructura YOLO
Mueve imágenes de imgs_lapiz/ a dataset_lapiz/ con división train/val
"""

import os
import shutil
import random
from pathlib import Path

def organize_lapiz_dataset():
    """Organizar dataset de lápiz desde imgs_lapiz"""
    
    print("📂 Organizando dataset de lápiz...")
    
    # Rutas
    source_dir = Path("imgs_lapiz")
    target_dir = Path("dataset_lapiz")
    
    # Verificar si existe la carpeta source
    if not source_dir.exists():
        print(f"❌ No se encontró la carpeta: {source_dir}")
        print("💡 Asegúrate de que las imágenes estén en 'imgs_lapiz/'")
        return False
    
    # Obtener lista de imágenes
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = []
    
    for ext in image_extensions:
        images.extend(list(source_dir.glob(f"*{ext}")))
        images.extend(list(source_dir.glob(f"*{ext.upper()}")))
    
    if not images:
        print(f"❌ No se encontraron imágenes en {source_dir}")
        return False
    
    print(f"✅ Encontradas {len(images)} imágenes")
    
    # Crear estructura de directorios
    dirs_to_create = [
        target_dir / "images" / "train",
        target_dir / "images" / "val", 
        target_dir / "labels" / "train",
        target_dir / "labels" / "val"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Creado: {dir_path}")
    
    # Mezclar imágenes aleatoriamente
    random.shuffle(images)
    
    # Dividir 70% train, 30% val
    train_split = int(len(images) * 0.7)
    train_images = images[:train_split]
    val_images = images[train_split:]
    
    print(f"📊 División del dataset:")
    print(f"   • Train: {len(train_images)} imágenes")
    print(f"   • Val: {len(val_images)} imágenes")
    
    # Copiar imágenes
    def copy_images(image_list, split_name):
        for img_path in image_list:
            # Destino para imagen
            dest_img = target_dir / "images" / split_name / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Crear archivo de etiqueta vacío (lo editaremos después)
            label_name = img_path.stem + ".txt"
            dest_label = target_dir / "labels" / split_name / label_name
            
            # Crear etiqueta de ejemplo (deberás editarla)
            with open(dest_label, 'w') as f:
                f.write("# Editar esta etiqueta con formato YOLO\n")
                f.write("# Formato: class_id center_x center_y width height\n")
                f.write("# Ejemplo: 0 0.5 0.5 0.3 0.4\n")
    
    copy_images(train_images, "train")
    copy_images(val_images, "val")
    
    # Crear data.yaml
    yaml_content = f"""# Dataset de Lápices
path: {target_dir.absolute()}
train: images/train
val: images/val

# Número de clases
nc: 1

# Nombres de clases  
names:
  0: lapiz
"""
    
    yaml_path = target_dir / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✅ Configuración creada: {yaml_path}")
    
    # Crear guía de etiquetado
    guide_content = f"""# 🏷️ GUÍA DE ETIQUETADO PARA LÁPICES

## ⚠️ IMPORTANTE: Debes etiquetar manualmente todas las imágenes

### 📋 Pasos siguientes:

1. **Instalar LabelImg:**
   ```bash
   pip install labelImg
   labelImg
   ```

2. **Configurar LabelImg:**
   - Open Dir: Selecciona `{target_dir}/images/train/`
   - Change Save Dir: Selecciona `{target_dir}/labels/train/`
   - Formato: YOLO
   - Clase: lapiz (id=0)

3. **Etiquetar:**
   - Dibuja un rectángulo alrededor de cada lápiz
   - Asigna clase "lapiz" (id=0)  
   - Guarda (Ctrl+S)
   - Siguiente imagen (D)

4. **Repetir para validación:**
   - Open Dir: `{target_dir}/images/val/`
   - Change Save Dir: `{target_dir}/labels/val/`

### 📐 Formato de etiqueta YOLO:
```
# Archivo: imagen123.txt
0 0.5 0.3 0.2 0.4
# 0 = clase (lapiz)
# 0.5 = centro X (normalizado 0-1)
# 0.3 = centro Y (normalizado 0-1) 
# 0.2 = ancho (normalizado 0-1)
# 0.4 = alto (normalizado 0-1)
```

### ✅ Verificación:
- Cada imagen.jpg debe tener su imagen.txt
- Todas las coordenadas entre 0 y 1
- Al menos {len(train_images)} archivos .txt en train/
- Al menos {len(val_images)} archivos .txt en val/

### 🚀 Después del etiquetado:
```bash
python train_lapiz_model.py
```
"""
    
    guide_path = target_dir / "ETIQUETADO_GUIA.md"
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"📖 Guía creada: {guide_path}")
    print("\n🎉 ¡Dataset organizado correctamente!")
    print(f"📂 Estructura creada en: {target_dir}")
    print("\n📋 PRÓXIMOS PASOS:")
    print("1. 🏷️  Etiquetar las imágenes con LabelImg")
    print("2. ✅ Verificar que cada imagen tenga su etiqueta")
    print("3. 🚀 Ejecutar entrenamiento")
    
    return True

def check_imgs_lapiz():
    """Verificar contenido de imgs_lapiz"""
    imgs_dir = Path("imgs_lapiz")
    
    if not imgs_dir.exists():
        print("❌ Carpeta 'imgs_lapiz' no encontrada")
        return
    
    # Contar archivos
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    total_images = 0
    
    print(f"📂 Contenido de {imgs_dir}:")
    for ext in image_extensions:
        count = len(list(imgs_dir.glob(f"*{ext}"))) + len(list(imgs_dir.glob(f"*{ext.upper()}")))
        if count > 0:
            print(f"   • {ext.upper()}: {count} archivos")
            total_images += count
    
    print(f"\n📊 Total de imágenes: {total_images}")
    
    if total_images < 50:
        print("⚠️ Tienes pocas imágenes. Recomendado: 100+ para mejor modelo")
    elif total_images < 100:
        print("✅ Cantidad aceptable. Más imágenes = mejor modelo")
    else:
        print("🎉 ¡Excelente cantidad de imágenes!")
    
    return total_images

if __name__ == "__main__":
    print("🖊️ ORGANIZADOR DE DATASET DE LÁPIZ")
    print("="*50)
    
    # Verificar imgs_lapiz
    total = check_imgs_lapiz()
    if total == 0:
        exit()
    
    print("\n" + "="*50)
    
    # Organizar dataset
    success = organize_lapiz_dataset()
    
    if success:
        print("\n💡 RECORDATORIO:")
        print("   Las etiquetas están vacías, DEBES etiquetarlas manualmente")
        print("   Usa LabelImg o Roboflow para etiquetar los lápices")
    else:
        print("\n❌ No se pudo organizar el dataset")