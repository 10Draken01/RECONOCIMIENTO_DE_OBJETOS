
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
