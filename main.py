

import cv2
import numpy as np
import pandas as pd
import gradio as gr
from ultralytics import YOLO
import tempfile
import os
import json
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import torch

class YOLOObjectDetector:
    """
    Clase principal para la detección de objetos usando YOLO
    """
    
    def __init__(self):
        """Inicializa los modelos YOLO"""
        # Modelo preentrenado (COCO dataset)
        self.pretrained_model = YOLO('yolov8n.pt')  # Usa yolov8s.pt o yolov8m.pt para mejor precisión
        
        # Modelo personalizado (se cargará cuando esté disponible)
        self.custom_model = None
        self.custom_model_path = "models/senales_alto.pt"  # Ruta del modelo personalizado
        
        # Clases objetivo del modelo preentrenado (COCO)
        self.target_classes = {
            0: 'persona',      # person en COCO
            2: 'automovil'     # car en COCO
        }
        
        # Contadores
        self.reset_counters()
        
        # Configuración de colores para visualización
        self.colors = {
            'persona': (0, 255, 0),      # Verde
            'automovil': (255, 0, 0),    # Azul
            'señal_alto': (0, 0, 255)    # Rojo
        }
        
    def reset_counters(self):
        """Reinicia los contadores de objetos"""
        self.counters = {
            'persona': 0,
            'automovil': 0,
            'señal_alto': 0
        }
        
    def load_custom_model(self, model_path: str) -> bool:
        """
        Carga el modelo personalizado para señales de alto
        
        Args:
            model_path: Ruta al modelo personalizado
            
        Returns:
            bool: True si se cargó correctamente, False en caso contrario
        """
        try:
            if os.path.exists(model_path):
                self.custom_model = YOLO(model_path)
                print(f"Modelo personalizado cargado desde: {model_path}")
                return True
            else:
                print(f"No se encontró el modelo personalizado en: {model_path}")
                return False
        except Exception as e:
            print(f"Error al cargar modelo personalizado: {e}")
            return False
    
    def detect_objects_image(self, image: np.ndarray, confidence: float = 0.5) -> Tuple[np.ndarray, Dict]:
        """
        Detecta objetos en una imagen
        
        Args:
            image: Imagen de entrada
            confidence: Umbral de confianza para detecciones
            
        Returns:
            Tuple: (imagen_procesada, estadísticas)
        """
        self.reset_counters()
        processed_image = image.copy()
        detections_info = []
        
        # Detección con modelo preentrenado
        results_pretrained = self.pretrained_model(processed_image, conf=confidence)
        
        for result in results_pretrained:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Obtener coordenadas y clase
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Solo procesar clases objetivo
                    if cls in self.target_classes:
                        class_name = self.target_classes[cls]
                        self.counters[class_name] += 1
                        
                        # Dibujar bounding box
                        color = self.colors[class_name]
                        cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Etiqueta
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(processed_image, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Guardar información de detección
                        detections_info.append({
                            'clase': class_name,
                            'confianza': float(conf),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
        
        # Detección con modelo personalizado (señales de alto)
        if self.custom_model is not None:
            results_custom = self.custom_model(processed_image, conf=confidence)
            
            for result in results_custom:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        
                        self.counters['señal_alto'] += 1
                        
                        # Dibujar bounding box
                        color = self.colors['señal_alto']
                        cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Etiqueta
                        label = f"señal_alto: {conf:.2f}"
                        cv2.putText(processed_image, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Guardar información de detección
                        detections_info.append({
                            'clase': 'señal_alto',
                            'confianza': float(conf),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
        
        # Mostrar contadores en la imagen
        self._draw_counters(processed_image)
        
        # Estadísticas
        stats = {
            'contadores': self.counters.copy(),
            'total_detecciones': sum(self.counters.values()),
            'detecciones': detections_info,
            'timestamp': datetime.now().isoformat()
        }
        
        return processed_image, stats
    
    def detect_objects_video(self, video_path: str, confidence: float = 0.5, 
                           output_path: Optional[str] = None) -> str:
        """
        Procesa un video completo detectando objetos
        
        Args:
            video_path: Ruta del video de entrada
            confidence: Umbral de confianza
            output_path: Ruta del video de salida (opcional)
            
        Returns:
            str: Ruta del video procesado
        """
        cap = cv2.VideoCapture(video_path)
        
        # Propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Configurar escritor de video
        if output_path is None:
            # Guardar en carpeta del proyecto en lugar de temp
            output_path = f"video_procesado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_stats = {'contadores': {'persona': 0, 'automovil': 0, 'señal_alto': 0}}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Procesar frame
            processed_frame, stats = self.detect_objects_image(frame, confidence)
            
            # Actualizar estadísticas totales
            for key in total_stats['contadores']:
                total_stats['contadores'][key] = max(
                    total_stats['contadores'][key], 
                    stats['contadores'][key]
                )
            
            # Añadir información del frame
            cv2.putText(processed_frame, f"Frame: {frame_count}", (10, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(processed_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"Video procesado guardado en: {output_path}")
        print(f"Estadísticas finales: {total_stats}")
        
        return output_path
    
    def _draw_counters(self, image: np.ndarray):
        """Dibuja los contadores en la imagen"""
        y_offset = 30
        for i, (class_name, count) in enumerate(self.counters.items()):
            text = f"{class_name.capitalize()}: {count}"
            color = self.colors[class_name]
            cv2.putText(image, text, (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def export_data(self, stats: Dict, format_type: str = 'json') -> str:
        """
        Exporta los datos de detección
        
        Args:
            stats: Estadísticas de detección
            format_type: Formato de exportación ('json' o 'csv')
            
        Returns:
            str: Ruta del archivo exportado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == 'json':
            filename = f"detecciones_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        
        elif format_type == 'csv':
            filename = f"detecciones_{timestamp}.csv"
            
            # Crear DataFrame con las detecciones
            df_detections = pd.DataFrame(stats['detecciones'])
            
            # Crear DataFrame con los contadores
            df_counters = pd.DataFrame([stats['contadores']])
            
            # Guardar ambos en el mismo archivo CSV
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("RESUMEN DE CONTADORES\n")
                df_counters.to_csv(f, index=False)
                f.write("\nDETALLE DE DETECCIONES\n")
                df_detections.to_csv(f, index=False)
        
        return filename

# Instancia global del detector
detector = YOLOObjectDetector()

def process_image_interface(image, confidence, load_custom):
    """Interfaz para procesar imágenes"""
    if image is None:
        return None, "Por favor, carga una imagen.", "", ""
    
    # Cargar modelo personalizado si se solicita
    if load_custom and detector.custom_model is None:
        detector.load_custom_model(detector.custom_model_path)
    
    # Procesar imagen
    processed_image, stats = detector.detect_objects_image(image, confidence)
    
    # Crear resumen de texto
    summary = f"""
    **Resultados de Detección:**
    
    🧑 Personas detectadas: {stats['contadores']['persona']}
    🚗 Automóviles detectados: {stats['contadores']['automovil']}
    🛑 Señales de alto detectadas: {stats['contadores']['señal_alto']}
    
    📊 Total de objetos: {stats['total_detecciones']}
    ⏰ Procesado: {stats['timestamp'][:19]}
    """
    
    # Exportar datos
    json_file = detector.export_data(stats, 'json')
    csv_file = detector.export_data(stats, 'csv')
    
    return processed_image, summary, json_file, csv_file

def process_video_interface(video, confidence, load_custom):
    """Interfaz para procesar videos"""
    if video is None:
        return None, "Por favor, carga un video.", "", ""
    
    # Cargar modelo personalizado si se solicita
    if load_custom and detector.custom_model is None:
        detector.load_custom_model(detector.custom_model_path)
    
    # Procesar video
    output_video = detector.detect_objects_video(video, confidence)
    
    # Crear mensaje con la ubicación del archivo
    success_message = f"""
    ✅ **Video procesado exitosamente!**
    
    📁 **Archivo guardado en:** `{output_video}`
    
    🎬 **Para ver el video:**
    1. Ve a la carpeta del proyecto
    2. Busca el archivo: `{os.path.basename(output_video)}`
    3. Reproduce con cualquier reproductor de video
    
    📊 **El video contiene todas las detecciones con cajas y contadores**
    """
    
    # Intentar devolver el video, si falla, solo devolver el mensaje
    try:
        return output_video, success_message, "", ""
    except:
        return None, success_message, "", ""

# Crear interfaz con Gradio
def create_interface():
    """Crea la interfaz de Gradio"""
    
    with gr.Blocks(title="Detector de Objetos YOLO", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🎯 Aplicación de Reconocimiento de Objetos con YOLO
        
        **Detecta:**
        - 🧑 **Personas** (modelo preentrenado)
        - 🚗 **Automóviles** (modelo preentrenado)  
        - 🛑 **Señales de alto mexicanas** (modelo personalizado)
        
        ---
        """)
        
        with gr.Tab("📸 Procesar Imagen"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="numpy", label="Subir Imagen")
                    confidence_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.1,
                        label="Umbral de Confianza"
                    )
                    load_custom_checkbox = gr.Checkbox(
                        label="Cargar modelo personalizado (señales de alto)", 
                        value=True
                    )
                    process_image_btn = gr.Button("🔍 Procesar Imagen", variant="primary")
                
                with gr.Column():
                    image_output = gr.Image(label="Imagen Procesada")
                    results_text = gr.Markdown(label="Resultados")
            
            with gr.Row():
                json_download = gr.File(label="Descargar JSON")
                csv_download = gr.File(label="Descargar CSV")
            
            process_image_btn.click(
                fn=process_image_interface,
                inputs=[image_input, confidence_slider, load_custom_checkbox],
                outputs=[image_output, results_text, json_download, csv_download]
            )
        
        with gr.Tab("🎥 Procesar Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Subir Video")
                    video_confidence_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.1,
                        label="Umbral de Confianza"
                    )
                    video_load_custom_checkbox = gr.Checkbox(
                        label="Cargar modelo personalizado (señales de alto)", 
                        value=True
                    )
                    process_video_btn = gr.Button("🎬 Procesar Video", variant="primary")
                
                with gr.Column():
                    video_output = gr.Video(label="Video Procesado")
                    video_results_text = gr.Markdown(label="Resultados")
            
            process_video_btn.click(
                fn=process_video_interface,
                inputs=[video_input, video_confidence_slider, video_load_custom_checkbox],
                outputs=[video_output, video_results_text, gr.File(), gr.File()]
            )
        
        with gr.Tab("📋 Información del Proyecto"):
            gr.Markdown("""
            ## 📋 Especificaciones Técnicas
            
            ### 🤖 Modelos Utilizados:
            - **YOLOv8n**: Modelo preentrenado para detección general
            - **Modelo personalizado**: Entrenado específicamente para señales de alto mexicanas
            
            ### 🎯 Clases Detectadas:
            1. **Personas** (modelo preentrenado COCO)
            2. **Automóviles** (modelo preentrenado COCO)
            3. **Señales de alto mexicanas** (modelo personalizado)
            
            ### 📊 Funcionalidades:
            - ✅ Detección en imágenes y videos
            - ✅ Conteo automático de objetos
            - ✅ Visualización con bounding boxes
            - ✅ Exportación de datos (JSON/CSV)
            - ✅ Interfaz web interactiva
            
            ### 🔧 Requisitos del Sistema:
            ```
            ultralytics>=8.0.0
            opencv-python>=4.5.0
            gradio>=3.0.0
            numpy>=1.21.0
            pandas>=1.3.0
            torch>=1.9.0
            ```
            
            ---
            **Desarrollado para el proyecto final de Visión por Computadora**
            """)
    
    return interface

if __name__ == "__main__":
    # Verificar disponibilidad de GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Usando CPU")
    
    # Intentar cargar modelo personalizado
    detector.load_custom_model(detector.custom_model_path)
    
    # Crear y lanzar interfaz
    interface = create_interface()
    interface.launch(
        share=False,  # Sin enlace público (evita problema del antivirus)
        server_name="127.0.0.1",  # Solo localhost
        server_port=7860,
        show_error=True,
        inbrowser=True  # Abrir automáticamente en el navegador
    )