
# Ejemplo 1: Uso básico de la aplicación
from app import YOLOObjectDetector
import cv2

# Crear detector
detector = YOLOObjectDetector()

# Cargar imagen
image = cv2.imread("test_images/ejemplo_1.jpg")

# Procesar imagen
result_image, results_json, stats = detector.process_image(image)

# Mostrar resultados
print(stats)

# Ejemplo 2: Entrenar modelo personalizado
from train_custom_model import CustomYOLOTrainer

# Crear entrenador
trainer = CustomYOLOTrainer("mi_detector")

# Definir clases personalizadas
classes = ["logo_upchiapas", "producto_especial"]
trainer.create_dataset_yaml(classes)

# Entrenar (después de preparar dataset)
# trainer.train_model()

# Ejemplo 3: Usar solo modelo preentrenado
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model('test_images/ejemplo_1.jpg')

for result in results:
    result.show()  # Mostrar imagen con detecciones
