
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
