"""
YOLO Detector Application - DETECCIÓN INTELIGENTE DE ESCRITURA
Sistema Híbrido:
- Modelo preentrenado: Detecta LIBROS (book)
- Modelo personalizado: Detecta LÁPICES
- Lógica inteligente: Cuando lápiz + libro están JUNTOS = KIT DE ESCRITURA
"""

import sys
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import threading
import time
import os
import math
import random

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                              QFileDialog, QSlider, QCheckBox, QTabWidget,
                              QTextEdit, QProgressBar, QFrame, QSizePolicy,
                              QSpacerItem, QScrollArea, QGroupBox)
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon

from ultralytics import YOLO
import torch

# ============================================================================
# DETECTOR HÍBRIDO DE ESCRITURA
# ============================================================================

class YOLOEscrituraHybridDetector:
    """
    Detector híbrido de escritura:
    - Usa modelo preentrenado para detectar libros
    - Usa modelo personalizado para detectar lápices  
    - Analiza proximidad para identificar kits de escritura
    """
    
    def __init__(self):
        print("✏️📚 Inicializando Detector Híbrido de Escritura...")
        
        # Modelos
        self.pretrained_model = None  # YOLOv8 preentrenado (para libros)
        self.lapiz_model = None       # Modelo personalizado (para lápices)
        
        # Configuración
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.proximity_threshold = 150  # Pixeles para considerar objetos "juntos"
        
        # Clases del modelo preentrenado (COCO)
        self.coco_classes = {
            74: 'book',  # Libro en COCO dataset
            # Agregar más clases si necesitas
        }
        
        # Colores para visualización
        self.colors = {
            'lapiz': (0, 255, 0),           # Verde
            'book': (255, 0, 0),            # Azul  
            'kit_escritura': (0, 0, 255),   # Rojo
            'default': (255, 255, 255)      # Blanco
        }
        
        # Cargar modelo preentrenado
        self._load_pretrained_model()
        self.reset_statistics()
        
        print("✅ Detector Híbrido listo")

    def reset_statistics(self):
        """Reiniciar estadísticas"""
        self.statistics = {
            'total_objects': 0,
            'objects_by_class': {},
            'kits_detected': 0,
            'analysis_timestamp': None
        }
    
    def _load_pretrained_model(self):
        """Cargar modelo YOLO preentrenado"""
        try:
            model_path = "yolov8n.pt"
            if os.path.exists(model_path):
                self.pretrained_model = YOLO(model_path)
                print(f"✅ Modelo preentrenado cargado: {model_path}")
            else:
                self.pretrained_model = YOLO('yolov8n.pt')
                print("✅ Modelo preentrenado descargado y cargado")
        except Exception as e:
            print(f"❌ Error cargando modelo preentrenado: {e}")
    
    def load_lapiz_model(self, model_path):
        """Cargar modelo personalizado de lápiz"""
        try:
            if os.path.exists(model_path):
                self.lapiz_model = YOLO(model_path)
                print(f"✅ Modelo de lápiz cargado: {model_path}")
                return True
            else:
                print(f"❌ No se encontró el modelo de lápiz: {model_path}")
                return False
        except Exception as e:
            print(f"❌ Error cargando modelo de lápiz: {e}")
            return False
    
    def detect_objects(self, image):
        """Detectar objetos usando ambos modelos y analizar proximidad"""
        detections = []
        
        # Detección con modelo preentrenado (libros)
        if self.pretrained_model:
            book_detections = self._detect_books(image)
            detections.extend(book_detections)
        
        # Detección con modelo personalizado (lápices)
        if self.lapiz_model:
            lapiz_detections = self._detect_lapices(image)
            detections.extend(lapiz_detections)
        
        # Analizar proximidad para detectar kits
        kit_detections = self._analyze_kits(detections)
        detections.extend(kit_detections)
        
        return detections
    
    def _detect_books(self, image):
        """Detectar libros usando modelo preentrenado"""
        detections = []
        
        try:
            results = self.pretrained_model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        # Filtrar solo libros
                        if class_id in self.coco_classes:
                            detection = {
                                'clase': 'book',
                                'clase_nombre': 'Libro',
                                'confianza': confidence,
                                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                                'centro': [int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2)],
                                'modelo': 'preentrenado',
                                'class_id': class_id,
                                'tipo': 'individual'
                            }
                            detections.append(detection)
        
        except Exception as e:
            print(f"❌ Error detectando libros: {e}")
        
        return detections
    
    def _detect_lapices(self, image):
        """Detectar lápices usando modelo personalizado"""
        detections = []
        
        try:
            results = self.lapiz_model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        detection = {
                            'clase': 'lapiz',
                            'clase_nombre': 'Lápiz',
                            'confianza': confidence,
                            'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                            'centro': [int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2)],
                            'modelo': 'personalizado',
                            'class_id': class_id,
                            'tipo': 'individual'
                        }
                        detections.append(detection)
        
        except Exception as e:
            print(f"❌ Error detectando lápices: {e}")
        
        return detections
    
    def _analyze_kits(self, detections):
        """Analizar proximidad entre lápices y libros para detectar kits"""
        kit_detections = []
        
        # Separar detecciones por tipo
        lapices = [d for d in detections if d['clase'] == 'lapiz']
        libros = [d for d in detections if d['clase'] == 'book']
        
        # Buscar pares lápiz-libro cercanos
        for lapiz in lapices:
            for libro in libros:
                # Calcular distancia entre centros
                dist = math.sqrt(
                    (lapiz['centro'][0] - libro['centro'][0])**2 + 
                    (lapiz['centro'][1] - libro['centro'][1])**2
                )
                
                # Si están lo suficientemente cerca, crear kit
                if dist <= self.proximity_threshold:
                    # Calcular bbox que englobe ambos objetos
                    x1 = min(lapiz['bbox'][0], libro['bbox'][0])
                    y1 = min(lapiz['bbox'][1], libro['bbox'][1])
                    x2 = max(lapiz['bbox'][2], libro['bbox'][2])
                    y2 = max(lapiz['bbox'][3], libro['bbox'][3])
                    
                    # Calcular confianza promedio
                    confidence = (lapiz['confianza'] + libro['confianza']) / 2
                    
                    kit_detection = {
                        'clase': 'kit_escritura',
                        'clase_nombre': 'Kit de Escritura',
                        'confianza': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'centro': [int((x1 + x2)/2), int((y1 + y2)/2)],
                        'modelo': 'híbrido',
                        'class_id': 999,  # ID especial para kit
                        'tipo': 'kit',
                        'componentes': {
                            'lapiz': lapiz,
                            'libro': libro
                        },
                        'distancia': round(dist, 1)
                    }
                    kit_detections.append(kit_detection)
        
        return kit_detections
    
    def draw_detections(self, image, detections):
        """Dibujar detecciones en la imagen"""
        annotated_image = image.copy()
        
        # Separar detecciones por tipo para dibujar en orden
        individuales = [d for d in detections if d['tipo'] == 'individual']
        kits = [d for d in detections if d['tipo'] == 'kit']
        
        # Dibujar detecciones individuales primero
        for detection in individuales:
            self._draw_single_detection(annotated_image, detection)
        
        # Dibujar kits después (más prominentes)
        for detection in kits:
            self._draw_kit_detection(annotated_image, detection)
        
        return annotated_image
    
    def _draw_single_detection(self, image, detection):
        """Dibujar una detección individual"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        color = self.colors.get(detection['clase'], self.colors['default'])
        
        # Dibujar rectángulo
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Preparar texto
        confidence = detection['confianza']
        clase_nombre = detection['clase_nombre']
        label = f"{clase_nombre} ({confidence:.2f})"
        
        # Dibujar texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(image, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   font, font_scale, (255, 255, 255), thickness)
    
    def _draw_kit_detection(self, image, detection):
        """Dibujar detección de kit (más prominente)"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        color = self.colors['kit_escritura']
        
        # Dibujar rectángulo más grueso para kit
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        
        # Preparar texto del kit
        confidence = detection['confianza']
        distancia = detection['distancia']
        label = f"KIT DE ESCRITURA ({confidence:.2f})"
        sublabel = f"Distancia: {distancia}px"
        
        # Dibujar texto principal
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        (sub_width, sub_height), _ = cv2.getTextSize(sublabel, font, 0.5, 1)
        
        # Fondo del texto
        cv2.rectangle(image, (x1, y1 - text_height - sub_height - 15), 
                     (x1 + max(text_width, sub_width), y1), color, -1)
        
        # Texto principal
        cv2.putText(image, label, (x1, y1 - sub_height - 8), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Subtexto
        cv2.putText(image, sublabel, (x1, y1 - 2), 
                   font, 0.5, (255, 255, 255), 1)
        
        # Dibujar línea conectando componentes
        lapiz_centro = detection['componentes']['lapiz']['centro']
        libro_centro = detection['componentes']['libro']['centro']
        cv2.line(image, tuple(lapiz_centro), tuple(libro_centro), color, 2)
        
        # Dibujar puntos en los centros
        cv2.circle(image, tuple(lapiz_centro), 5, color, -1)
        cv2.circle(image, tuple(libro_centro), 5, color, -1)
    
    def count_objects(self, detections):
        """Contar objetos por clase"""
        counts = {}
        
        for detection in detections:
            class_name = detection['clase_nombre']
            if class_name not in counts:
                counts[class_name] = 0
            counts[class_name] += 1
        
        return counts
    
    def analyze_writing_setup(self, detections):
        """Analizar configuración de escritura"""
        analysis = {
            'lapices_individuales': len([d for d in detections if d['clase'] == 'lapiz']),
            'libros_individuales': len([d for d in detections if d['clase'] == 'book']),
            'kits_detectados': len([d for d in detections if d['clase'] == 'kit_escritura']),
            'total_objetos': len(detections),
            'configuracion': 'incompleta'
        }
        
        # Determinar tipo de configuración
        if analysis['kits_detectados'] > 0:
            analysis['configuracion'] = 'kit_completo'
        elif analysis['lapices_individuales'] > 0 and analysis['libros_individuales'] > 0:
            analysis['configuracion'] = 'elementos_separados'
        elif analysis['lapices_individuales'] > 0:
            analysis['configuracion'] = 'solo_lapices'
        elif analysis['libros_individuales'] > 0:
            analysis['configuracion'] = 'solo_libros'
        
        return analysis
    
    def set_confidence_threshold(self, threshold):
        """Establecer umbral de confianza"""
        self.confidence_threshold = max(0.1, min(1.0, threshold))
    
    def set_iou_threshold(self, threshold):
        """Establecer umbral de IoU"""
        self.iou_threshold = max(0.1, min(1.0, threshold))
    
    def set_proximity_threshold(self, threshold):
        """Establecer umbral de proximidad para kits"""
        self.proximity_threshold = max(50, min(500, threshold))

# ============================================================================
# HILO PARA PROCESAMIENTO DE VIDEO
# ============================================================================

class VideoProcessorThread(QThread):
    """Hilo para procesar video en tiempo real"""
    
    frame_ready = Signal(np.ndarray, dict, dict)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.detector = YOLOEscrituraHybridDetector()
        self.running = False
        self.source = None
        self.frame_skip = 1
        self.frame_count = 0
    
    def set_source(self, source):
        """Establecer fuente de video"""
        self.source = source
    
    def start_processing(self):
        """Iniciar procesamiento"""
        self.running = True
        self.start()
    
    def stop_processing(self):
        """Detener procesamiento"""
        self.running = False
        self.wait()
    
    def run(self):
        """Ejecutar procesamiento de video"""
        if self.source is None:
            self.error_occurred.emit("No se ha establecido fuente de video")
            return
        
        # Abrir fuente de video
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(str(self.source))
        
        if not cap.isOpened():
            self.error_occurred.emit(f"No se pudo abrir la fuente: {self.source}")
            return
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                if self.frame_count % self.frame_skip != 0:
                    continue
                
                # Detectar objetos
                detections = self.detector.detect_objects(frame)
                
                # Contar objetos
                counts = self.detector.count_objects(detections)
                
                # Analizar configuración
                setup_analysis = self.detector.analyze_writing_setup(detections)
                
                # Dibujar detecciones
                annotated_frame = self.detector.draw_detections(frame, detections)
                
                # Emitir frame procesado
                self.frame_ready.emit(annotated_frame, counts, setup_analysis)
                
                self.msleep(30)
        
        except Exception as e:
            self.error_occurred.emit(f"Error procesando video: {e}")
        
        finally:
            cap.release()

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

class YOLOEscrituraHybridApp(QMainWindow):
    """Aplicación principal del detector híbrido de escritura"""
    
    def __init__(self):
        super().__init__()
        self.detector = YOLOEscrituraHybridDetector()
        self.video_processor = VideoProcessorThread()
        self.current_image = None
        self.current_detections = []
        
        self.init_ui()
        self.connect_signals()
        self.apply_dark_theme()
    
    def init_ui(self):
        """Inicializar interfaz de usuario"""
        self.setWindowTitle("✏️📚 YOLO Detector Híbrido - Lápiz + Libro = Kit de Escritura")
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(central_widget)
        
        # Panel izquierdo - Controles
        self.create_control_panel(main_layout)
        
        # Panel derecho - Visualización
        self.create_display_panel(main_layout)
    
    def create_control_panel(self, main_layout):
        """Crear panel de controles"""
        control_frame = QFrame()
        control_frame.setFixedWidth(380)
        control_frame.setFrameStyle(QFrame.StyledPanel)
        
        control_layout = QVBoxLayout(control_frame)
        
        # Título
        title = QLabel("✏️📚 DETECTOR HÍBRIDO")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ff88; margin: 10px;")
        title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title)
        
        subtitle = QLabel("Lápiz + Libro = Kit")
        subtitle.setStyleSheet("font-size: 12px; color: #888; margin-bottom: 10px;")
        subtitle.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(subtitle)
        
        # Sección de archivos
        files_group = QGroupBox("📁 Archivos")
        files_layout = QVBoxLayout(files_group)
        
        self.btn_load_image = QPushButton("Cargar Imagen")
        self.btn_load_video = QPushButton("Cargar Video")
        self.btn_camera = QPushButton("Usar Cámara")
        
        files_layout.addWidget(self.btn_load_image)
        files_layout.addWidget(self.btn_load_video)
        files_layout.addWidget(self.btn_camera)
        
        control_layout.addWidget(files_group)
        
        # Sección de modelos
        models_group = QGroupBox("🤖 Modelos")
        models_layout = QVBoxLayout(models_group)
        
        # Estado de modelos
        self.pretrained_status = QLabel("📚 Libros: ✅ Modelo preentrenado")
        self.pretrained_status.setStyleSheet("color: #00ff88; padding: 2px;")
        models_layout.addWidget(self.pretrained_status)
        
        self.lapiz_status = QLabel("✏️ Lápiz: ❌ Sin modelo")
        self.lapiz_status.setStyleSheet("color: #ff6666; padding: 2px;")
        models_layout.addWidget(self.lapiz_status)
        
        self.btn_load_lapiz_model = QPushButton("Cargar Modelo Lápiz")
        models_layout.addWidget(self.btn_load_lapiz_model)
        
        control_layout.addWidget(models_group)
        
        # Sección de configuración
        config_group = QGroupBox("⚙️ Configuración")
        config_layout = QVBoxLayout(config_group)
        
        # Umbral de confianza
        config_layout.addWidget(QLabel("Confianza:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 100)
        self.confidence_slider.setValue(50)
        self.confidence_label = QLabel("0.50")
        
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(self.confidence_slider)
        conf_layout.addWidget(self.confidence_label)
        config_layout.addLayout(conf_layout)
        
        # Umbral de proximidad para kits
        config_layout.addWidget(QLabel("Proximidad Kit (px):"))
        self.proximity_slider = QSlider(Qt.Horizontal)
        self.proximity_slider.setRange(50, 300)
        self.proximity_slider.setValue(150)
        self.proximity_label = QLabel("150")
        
        prox_layout = QHBoxLayout()
        prox_layout.addWidget(self.proximity_slider)
        prox_layout.addWidget(self.proximity_label)
        config_layout.addLayout(prox_layout)
        
        control_layout.addWidget(config_group)
        
        # Sección de estadísticas
        stats_group = QGroupBox("📊 Análisis de Escritura")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(280)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        control_layout.addWidget(stats_group)
        
        # Botones de control
        control_buttons_group = QGroupBox("🎮 Control")
        control_buttons_layout = QVBoxLayout(control_buttons_group)
        
        self.btn_start_stop = QPushButton("▶️ Iniciar")
        self.btn_save_results = QPushButton("💾 Guardar Resultados")
        
        control_buttons_layout.addWidget(self.btn_start_stop)
        control_buttons_layout.addWidget(self.btn_save_results)
        
        control_layout.addWidget(control_buttons_group)
        
        # Espaciador
        control_layout.addStretch()
        
        main_layout.addWidget(control_frame)
    
    def create_display_panel(self, main_layout):
        """Crear panel de visualización"""
        display_frame = QFrame()
        display_frame.setFrameStyle(QFrame.StyledPanel)
        
        display_layout = QVBoxLayout(display_frame)
        
        # Área de imagen/video
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 2px solid #333; background-color: #1a1a1a;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Cargar imagen o video\\n\\n✏️ Detecta LÁPICES (modelo personalizado)\\n📚 Detecta LIBROS (modelo preentrenado)\\n🎯 Cuando están JUNTOS = KIT DE ESCRITURA")
        
        display_layout.addWidget(self.image_label)
        
        # Barra de estado
        self.status_label = QLabel("Listo - Carga el modelo de lápiz para comenzar")
        self.status_label.setStyleSheet("color: #00ff88; padding: 5px;")
        display_layout.addWidget(self.status_label)
        
        main_layout.addWidget(display_frame)
    
    def connect_signals(self):
        """Conectar señales"""
        # Botones de archivos
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_camera.clicked.connect(self.start_camera)
        
        # Botones de modelos
        self.btn_load_lapiz_model.clicked.connect(self.load_lapiz_model)
        
        # Botones de control
        self.btn_start_stop.clicked.connect(self.toggle_processing)
        self.btn_save_results.clicked.connect(self.save_results)
        
        # Sliders
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        self.proximity_slider.valueChanged.connect(self.update_proximity)
        
        # Video processor
        self.video_processor.frame_ready.connect(self.display_frame)
        self.video_processor.error_occurred.connect(self.show_error)
    
    def apply_dark_theme(self):
        """Aplicar tema oscuro"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #404040;
                border: 2px solid #555;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #505050;
                border-color: #00ff88;
            }
            QPushButton:pressed {
                background-color: #303030;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #1a1a1a;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
            }
            QLabel {
                color: #ffffff;
            }
        """)
    
    def load_image(self):
        """Cargar imagen"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Imagen", "", 
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.current_image = image
                self.process_image(image)
                self.status_label.setText(f"Imagen cargada: {Path(file_path).name}")
            else:
                self.show_error("No se pudo cargar la imagen")
    
    def load_video(self):
        """Cargar video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Video", "", 
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        
        if file_path:
            self.video_processor.set_source(file_path)
            self.status_label.setText(f"Video cargado: {Path(file_path).name}")
            self.btn_start_stop.setText("▶️ Iniciar Video")
    
    def start_camera(self):
        """Iniciar cámara"""
        self.video_processor.set_source(0)
        self.status_label.setText("Cámara configurada")
        self.btn_start_stop.setText("▶️ Iniciar Cámara")
    
    def load_lapiz_model(self):
        """Cargar modelo de lápiz"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Modelo de Lápiz", "", 
            "Modelos YOLO (*.pt *.onnx)"
        )
        
        if file_path:
            success = self.detector.load_lapiz_model(file_path)
            self.video_processor.detector.load_lapiz_model(file_path)
            
            if success:
                self.lapiz_status.setText("✏️ Lápiz: ✅ Modelo cargado")
                self.lapiz_status.setStyleSheet("color: #00ff88; padding: 2px;")
                self.status_label.setText(f"Modelo de lápiz cargado: {Path(file_path).name}")
            else:
                self.show_error("No se pudo cargar el modelo de lápiz")
    
    def process_image(self, image):
        """Procesar imagen individual"""
        detections = self.detector.detect_objects(image)
        self.current_detections = detections
        
        annotated_image = self.detector.draw_detections(image, detections)
        self.display_image(annotated_image)
        
        counts = self.detector.count_objects(detections)
        setup_analysis = self.detector.analyze_writing_setup(detections)
        self.update_statistics(counts, detections, setup_analysis)
    
    def toggle_processing(self):
        """Alternar procesamiento de video"""
        if not self.video_processor.running:
            self.video_processor.start_processing()
            self.btn_start_stop.setText("⏹️ Detener")
        else:
            self.video_processor.stop_processing()
            self.btn_start_stop.setText("▶️ Iniciar")
    
    def display_frame(self, frame, counts, setup_analysis):
        """Mostrar frame de video"""
        self.display_image(frame)
        self.update_statistics(counts, [], setup_analysis)
    
    def display_image(self, image):
        """Mostrar imagen en la interfaz"""
        height, width, channel = image.shape
        label_size = self.image_label.size()
        
        scale_w = label_size.width() / width
        scale_h = label_size.height() / height
        scale = min(scale_w, scale_h, 1.0)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized_image = cv2.resize(image, (new_width, new_height))
        
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
    
    def update_statistics(self, counts, detections, setup_analysis):
        """Actualizar estadísticas"""
        stats_text = "📊 ANÁLISIS DE ESCRITURA\\n\\n"
        
        # Conteos por clase
        total_objects = sum(counts.values())
        stats_text += f"Total de objetos: {total_objects}\\n\\n"
        
        for class_name, count in counts.items():
            if class_name == "Kit de Escritura":
                stats_text += f"🎯 {class_name}: {count}\\n"
            elif class_name == "Lápiz":
                stats_text += f"✏️ {class_name}: {count}\\n"
            elif class_name == "Libro":
                stats_text += f"📚 {class_name}: {count}\\n"
            else:
                stats_text += f"• {class_name}: {count}\\n"
        
        # Análisis de configuración
        if setup_analysis:
            stats_text += "\\n🔍 CONFIGURACIÓN DETECTADA:\\n"
            config = setup_analysis['configuracion']
            
            if config == 'kit_completo':
                stats_text += "✅ Kit de escritura completo\\n"
            elif config == 'elementos_separados':
                stats_text += "⚠️ Elementos separados (no forman kit)\\n"
            elif config == 'solo_lapices':
                stats_text += "📝 Solo lápices detectados\\n"
            elif config == 'solo_libros':
                stats_text += "📖 Solo libros detectados\\n"
            else:
                stats_text += "❌ Configuración incompleta\\n"
            
            stats_text += f"\\n📋 DETALLES:\\n"
            stats_text += f"   • Lápices individuales: {setup_analysis['lapices_individuales']}\\n"
            stats_text += f"   • Libros individuales: {setup_analysis['libros_individuales']}\\n"
            stats_text += f"   • Kits detectados: {setup_analysis['kits_detectados']}\\n"
        
        if detections:
            stats_text += "\\n🔎 DETECCIONES DETALLADAS:\\n"
            for i, detection in enumerate(detections[:6]):
                clase = detection["clase_nombre"]
                conf = detection["confianza"]
                
                if detection.get('tipo') == 'kit':
                    dist = detection.get('distancia', 0)
                    stats_text += f"{i+1}. 🎯 {clase} ({conf:.2f}) - {dist}px\\n"
                else:
                    modelo = detection["modelo"]
                    stats_text += f"{i+1}. {clase} ({conf:.2f}) [{modelo}]\\n"
            
            if len(detections) > 6:
                stats_text += f"... y {len(detections) - 6} más\\n"
        
        self.stats_text.setText(stats_text)
    
    def update_confidence(self, value):
        """Actualizar umbral de confianza"""
        threshold = value / 100.0
        self.confidence_label.setText(f"{threshold:.2f}")
        self.detector.set_confidence_threshold(threshold)
        self.video_processor.detector.set_confidence_threshold(threshold)
    
    def update_proximity(self, value):
        """Actualizar umbral de proximidad"""
        self.proximity_label.setText(str(value))
        self.detector.set_proximity_threshold(value)
        self.video_processor.detector.set_proximity_threshold(value)
    
    def save_results(self):
        """Guardar resultados"""
        if self.current_image is None and not self.current_detections:
            self.show_error("No hay resultados para guardar")
            return
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.current_image is not None:
            annotated_image = self.detector.draw_detections(self.current_image, self.current_detections)
            output_path = output_dir / f"hybrid_detection_{timestamp}.jpg"
            cv2.imwrite(str(output_path), annotated_image)
        
        # Guardar resultados
        results = {
            "timestamp": timestamp,
            "detections": self.current_detections,
            "counts": self.detector.count_objects(self.current_detections),
            "setup_analysis": self.detector.analyze_writing_setup(self.current_detections),
            "proximity_threshold": self.detector.proximity_threshold
        }
        
        json_path = output_dir / f"hybrid_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.status_label.setText(f"Resultados guardados en: {output_dir}")
    
    def show_error(self, message):
        """Mostrar mensaje de error"""
        self.status_label.setText(f"❌ Error: {message}")
        print(f"Error: {message}")

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    app = QApplication(sys.argv)
    
    app.setApplicationName("YOLO Detector Híbrido - Escritura")
    app.setApplicationVersion("2.0")
    
    window = YOLOEscrituraHybridApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()