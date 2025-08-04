
# Configuración del Proyecto YOLO

## Modelos Disponibles
- yolov8n.pt: Modelo nano (más rápido)
- yolov8s.pt: Modelo small (balance)  
- yolov8m.pt: Modelo medium (más preciso)
- yolov8l.pt: Modelo large (muy preciso)
- yolov8x.pt: Modelo extra large (máxima precisión)

## Clases del Modelo Preentrenado (COCO)
0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 5: bus,
6: train, 7: truck, 8: boat, 9: traffic light, 10: fire hydrant,
[... 80 clases total]

## Configuración de Entrenamiento Recomendada
- Epochs: 100-300 (según dataset)
- Batch size: 16 (ajustar según GPU)
- Image size: 640x640
- Learning rate: 0.01
