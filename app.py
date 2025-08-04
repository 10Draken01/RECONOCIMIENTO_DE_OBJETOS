#!/usr/bin/env python3
"""
Aplicación de Detección COHERENTE con YOLO
Universidad Politécnica de Chiapas

ESTRATEGIA COHERENTE:
🟢 PREENTRENADO: botellas + tijeras (objetos individuales)
🔵 PERSONALIZADO: kit_reciclaje (botellas + tijeras juntas para manualidades)

CONCEPTO:
- Detecta botellas y tijeras por separado
- Detecta "kit de reciclaje" cuando ambos están presentes para manualidades
"""

import cv2
import numpy as np
import pandas as pd
import gradio as gr
from ultralytics import YOLO
import json
import os
import tempfile
from datetime import datetime
from typing import Tuple, Dict
import time

class DetectorReciclaje:
    """
    Detector coherente: objetos individuales + kit de reciclaje.
    
    DETECTA:
    🍼 Botellas individuales
    ✂️ Tijeras individuales  
    ♻️ Kit de reciclaje (botella + tijeras juntas para manualidades)
    """
    
    def __init__(self):
        print("♻️ Inicializando Detector de Reciclaje Creativo...")
        
        # Modelos
        self.pretrained_model = None
        self.custom_model = None
        
        # Objetos individuales que detecta YOLO
        self.target_classes = ['bottle', 'scissors']
        
        # Emojis para visualización
        self.emojis = {
            'bottle': '🍼',
            'scissors': '✂️', 
            'kit_reciclaje': '♻️'
        }
        
        # Colores muy visibles
        self.colors = {
            'pretrained': (0, 255, 0),    # Verde para objetos individuales
            'custom': (255, 0, 0),        # Azul para kit de reciclaje
            'text': (255, 255, 255),      # Blanco
            'bg': (0, 0, 0)               # Negro
        }
        
        # Resultados
        self.results = {'pretrained': {}, 'custom': {}}
        
        # Cargar modelos
        self.load_models()
    
    def load_models(self):
        """Carga modelos con enfoque en reciclaje."""
        try:
            print("📥 Cargando modelo YOLO preentrenado...")
            self.pretrained_model = YOLO('yolov8n.pt')
            print("✅ Modelo YOLO listo")
            
            print("🎯 ESTRATEGIA DE DETECCIÓN:")
            print("   🍼 Botellas individuales (para reciclar)")
            print("   ✂️ Tijeras individuales (para cortar)")
            print("   ♻️ Kit de reciclaje (botella + tijeras juntas)")
            
            # Modelo personalizado para kit de reciclaje
            custom_path = 'models/kit_reciclaje_model.pt'
            if os.path.exists(custom_path):
                self.custom_model = YOLO(custom_path)
                print("✅ Modelo kit de reciclaje cargado")
            else:
                print("⏳ Modelo kit de reciclaje pendiente")
                print("   ♻️ Para detectar kits: entrena con imágenes de botella+tijeras")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def detect_pretrained(self, image, confidence=0.25):
        """Detecta botellas y tijeras individuales."""
        if self.pretrained_model is None:
            return image, {}
        
        results = self.pretrained_model(image, conf=confidence, verbose=False)
        
        annotated = image.copy()
        counts = {}
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.pretrained_model.names[class_id]
                    
                    # Solo nuestras clases objetivo
                    if class_name in self.target_classes:
                        counts[class_name] = counts.get(class_name, 0) + 1
                        
                        # CAJA VERDE MUY VISIBLE
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), 
                                    self.colors['pretrained'], 4)
                        
                        # ETIQUETA GRANDE con contexto de reciclaje
                        emoji = self.emojis.get(class_name, '📦')
                        if class_name == 'bottle':
                            label = f"{emoji} BOTELLA-RECICLABLE: {conf:.2f}"
                        elif class_name == 'scissors':
                            label = f"{emoji} TIJERAS-MANUALIDAD: {conf:.2f}"
                        else:
                            label = f"{emoji} {class_name.upper()}: {conf:.2f}"
                        
                        # Fondo sólido para etiqueta
                        font_scale = 0.8
                        thickness = 2
                        (text_w, text_h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        
                        # Posición de etiqueta (arriba del objeto)
                        label_y = y1 - 10 if y1 > 40 else y2 + 40
                        
                        # Fondo verde para etiqueta
                        cv2.rectangle(annotated, 
                                    (x1, label_y - text_h - 10), 
                                    (x1 + text_w + 10, label_y + 5),
                                    self.colors['pretrained'], -1)
                        
                        # Borde negro
                        cv2.rectangle(annotated, 
                                    (x1, label_y - text_h - 10), 
                                    (x1 + text_w + 10, label_y + 5),
                                    self.colors['bg'], 2)
                        
                        # Texto blanco
                        cv2.putText(annotated, label, (x1 + 5, label_y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                                  self.colors['text'], thickness)
        
        return annotated, counts
    
    def detect_custom(self, image, confidence=0.5):
        """Detecta kit de reciclaje (botella + tijeras juntas)."""
        if self.custom_model is None:
            return image, {}
        
        results = self.custom_model(image, conf=confidence, verbose=False)
        
        annotated = image.copy()
        counts = {}
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.custom_model.names[class_id]
                    
                    counts[class_name] = counts.get(class_name, 0) + 1
                    
                    # CAJA AZUL para kit de reciclaje
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), 
                                self.colors['custom'], 4)
                    
                    # Etiqueta especial para kit
                    label = f"♻️ KIT-RECICLAJE: {conf:.2f}"
                    
                    font_scale = 0.8
                    thickness = 2
                    (text_w, text_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    
                    label_y = y1 - 10 if y1 > 40 else y2 + 40
                    
                    # Fondo azul para kit de reciclaje
                    cv2.rectangle(annotated,
                                (x1, label_y - text_h - 10),
                                (x1 + text_w + 10, label_y + 5),
                                self.colors['custom'], -1)
                    
                    # Borde negro
                    cv2.rectangle(annotated,
                                (x1, label_y - text_h - 10),
                                (x1 + text_w + 10, label_y + 5),
                                self.colors['bg'], 2)
                    
                    # Texto blanco
                    cv2.putText(annotated, label, (x1 + 5, label_y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                              self.colors['text'], thickness)
        
        return annotated, counts
    
    def process_image(self, image, confidence=0.25):
        """Procesa imagen buscando objetos individuales y kits."""
        if image is None:
            return None, "❌ Sin imagen", "{}"
        
        start_time = time.time()
        
        # Convertir RGB a BGR
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Detectar objetos individuales
        img_with_pretrained, counts_pretrained = self.detect_pretrained(image_bgr, confidence)
        
        # Detectar kit de reciclaje
        img_final, counts_custom = self.detect_custom(img_with_pretrained, confidence)
        
        # Agregar información contextual
        height, width = img_final.shape[:2]
        
        # Análisis del contexto de reciclaje
        has_bottle = counts_pretrained.get('bottle', 0) > 0
        has_scissors = counts_pretrained.get('scissors', 0) > 0
        has_kit = counts_custom.get('kit_reciclaje', 0) > 0
        
        # Mensaje contextual
        if has_bottle and has_scissors and not has_kit:
            context_msg = "POTENCIAL KIT DE RECICLAJE DETECTADO!"
            color = (0, 255, 255)  # Amarillo
        elif has_kit:
            context_msg = "KIT DE RECICLAJE CONFIRMADO!"
            color = (255, 0, 0)    # Azul
        elif has_bottle or has_scissors:
            context_msg = "COMPONENTES DE RECICLAJE PRESENTES"
            color = (0, 255, 0)    # Verde
        else:
            context_msg = "BUSCAR: BOTELLAS Y TIJERAS"
            color = (128, 128, 128)  # Gris
        
        # Mostrar mensaje contextual
        cv2.rectangle(img_final, (10, 10), (width-10, 80), (0, 0, 0), -1)
        cv2.rectangle(img_final, (10, 10), (width-10, 80), color, 3)
        cv2.putText(img_final, context_msg, (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Contador total
        total = sum(counts_pretrained.values()) + sum(counts_custom.values())
        counter_text = f"OBJETOS DETECTADOS: {total}"
        cv2.putText(img_final, counter_text, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Guardar resultados
        processing_time = time.time() - start_time
        self.results = {
            'pretrained': counts_pretrained,
            'custom': counts_custom,
            'time': processing_time,
            'context': {
                'has_bottle': has_bottle,
                'has_scissors': has_scissors,
                'has_kit': has_kit,
                'potential_kit': has_bottle and has_scissors and not has_kit
            }
        }
        
        # Convertir a RGB para Gradio
        img_final_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
        
        # Generar estadísticas
        stats = self.generate_stats()
        json_results = json.dumps(self.results, indent=2)
        
        return img_final_rgb, stats, json_results
    
    def process_video(self, video_path, confidence=0.25, max_frames=100):
        """Procesa video buscando objetos y kits de reciclaje - VERSION CORREGIDA."""
        if not video_path or not os.path.exists(video_path):
            return None, "❌ Video no encontrado", "{}"
        
        print(f"🎬 Procesando video de reciclaje: {video_path}")
        
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "❌ No se puede abrir video", "{}"
        
        # Propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or fps > 60:  # Validar FPS
            fps = 25
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validar dimensiones
        if width <= 0 or height <= 0:
            cap.release()
            return None, "❌ Video con dimensiones inválidas", "{}"
        
        frames_to_process = min(total_frames, max_frames)
        
        print(f"📊 Video: {width}x{height}, {fps}fps, procesando {frames_to_process} frames")
        
        # CORRECCION: Usar formato más compatible
        output_filename = f"video_procesado_{int(time.time())}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        # CODEC más compatible para Gradio
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Cambio de mp4v a XVID
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Verificar que el writer se inicializó correctamente
        if not out.isOpened():
            print("⚠️ Intentando con codec alternativo...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path.replace('.mp4', '.avi'), fourcc, fps, (width, height))
            output_path = output_path.replace('.mp4', '.avi')
        
        if not out.isOpened():
            cap.release()
            return None, "❌ No se puede crear video de salida", "{}"
        
        # Contadores
        total_pretrained = {}
        total_custom = {}
        frame_count = 0
        potential_kits = 0
        
        start_time = time.time()
        
        try:
            print("🔄 Iniciando procesamiento frame por frame...")
            
            while frame_count < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️ No se pudo leer frame {frame_count + 1}")
                    break
                
                # VALIDAR FRAME
                if frame is None or frame.size == 0:
                    print(f"⚠️ Frame {frame_count + 1} está vacío")
                    continue
                
                # Procesar frame
                try:
                    frame_with_pretrained, counts_pretrained = self.detect_pretrained(frame, confidence)
                    frame_final, counts_custom = self.detect_custom(frame_with_pretrained, confidence)
                except Exception as e:
                    print(f"⚠️ Error procesando frame {frame_count + 1}: {e}")
                    frame_final = frame  # Usar frame original si hay error
                    counts_pretrained = {}
                    counts_custom = {}
                
                # Análisis del contexto por frame
                has_bottle = counts_pretrained.get('bottle', 0) > 0
                has_scissors = counts_pretrained.get('scissors', 0) > 0
                has_kit = counts_custom.get('kit_reciclaje', 0) > 0
                
                if has_bottle and has_scissors:
                    potential_kits += 1
                
                # Acumular conteos
                for cls, count in counts_pretrained.items():
                    total_pretrained[cls] = total_pretrained.get(cls, 0) + count
                for cls, count in counts_custom.items():
                    total_custom[cls] = total_custom.get(cls, 0) + count
                
                # MEJORAR INFORMACIÓN DEL FRAME
                current_detections = sum(counts_pretrained.values()) + sum(counts_custom.values())
                
                # Barra de información MÁS VISIBLE
                info_height = 120
                info_y = height - info_height
                
                # Fondo semi-transparente más grande
                overlay = frame_final.copy()
                cv2.rectangle(overlay, (0, info_y), (width, height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame_final, 0.3, 0, frame_final)
                
                # Información del frame
                frame_text = f"Frame: {frame_count+1}/{frames_to_process}"
                detect_text = f"Detecciones: {current_detections}"
                progress_text = f"Progreso: {((frame_count+1)/frames_to_process)*100:.1f}%"
                
                # Contexto de reciclaje
                if has_kit:
                    context_text = "♻️ KIT DE RECICLAJE DETECTADO!"
                    context_color = (255, 0, 0)  # Azul
                elif has_bottle and has_scissors:
                    context_text = "🔄 POTENCIAL KIT PRESENTE"
                    context_color = (0, 255, 255)  # Amarillo
                elif has_bottle:
                    context_text = "🍼 BOTELLA DETECTADA" 
                    context_color = (0, 255, 0)  # Verde
                elif has_scissors:
                    context_text = "✂️ TIJERAS DETECTADAS"
                    context_color = (0, 255, 0)  # Verde
                else:
                    context_text = "🔍 BUSCANDO OBJETOS RECICLABLES"
                    context_color = (128, 128, 128)  # Gris
                
                # Dibujar textos con mejor visibilidad
                y_offset = info_y + 25
                cv2.putText(frame_final, frame_text, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                y_offset += 25
                cv2.putText(frame_final, detect_text, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                y_offset += 25
                cv2.putText(frame_final, progress_text, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                y_offset += 25
                cv2.putText(frame_final, context_text, (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, context_color, 2)
                
                # VALIDAR FRAME ANTES DE ESCRIBIR
                if frame_final.shape == (height, width, 3):
                    out.write(frame_final)
                else:
                    print(f"⚠️ Frame {frame_count + 1} con dimensiones incorrectas")
                    # Redimensionar si es necesario
                    frame_resized = cv2.resize(frame_final, (width, height))
                    out.write(frame_resized)
                
                frame_count += 1
                
                # Progreso cada 10 frames
                if frame_count % 10 == 0:
                    progress = (frame_count / frames_to_process) * 100
                    print(f"📊 Procesando: {progress:.1f}% ({frame_count}/{frames_to_process})")
        
        except Exception as e:
            print(f"❌ Error durante procesamiento: {e}")
        
        finally:
            cap.release()
            out.release()
            print(f"🔚 Procesamiento completado: {frame_count} frames")
        
        # VERIFICAR QUE EL VIDEO SE CREÓ CORRECTAMENTE
        if not os.path.exists(output_path):
            return None, "❌ No se pudo crear el video de salida", "{}"
        
        # Verificar tamaño del archivo
        file_size = os.path.getsize(output_path)
        if file_size < 1000:  # Menos de 1KB indica problema
            return None, "❌ Video de salida demasiado pequeño (posible error)", "{}"
        
        processing_time = time.time() - start_time
        
        # Guardar resultados
        self.results = {
            'pretrained': total_pretrained,
            'custom': total_custom,
            'time': processing_time,
            'frames': frame_count,
            'recycling_analysis': {
                'potential_kit_frames': potential_kits,
                'kit_potential_percentage': (potential_kits / frame_count * 100) if frame_count > 0 else 0
            }
        }
        
        stats = self.generate_video_stats(frame_count, processing_time, potential_kits)
        json_results = json.dumps(self.results, indent=2)
        
        print(f"✅ Video procesado exitosamente: {output_path}")
        print(f"📊 Tamaño: {file_size/1024/1024:.1f} MB")
        
        return output_path, stats, json_results
    
    def generate_stats(self):
        """Genera estadísticas con enfoque en reciclaje."""
        stats = []
        stats.append("♻️ ANÁLISIS DE RECICLAJE CREATIVO")
        stats.append("="*60)
        
        # Contexto
        context = self.results.get('context', {})
        
        stats.append("\n🎯 ANÁLISIS CONTEXTUAL:")
        if context.get('has_kit'):
            stats.append("   ✅ KIT DE RECICLAJE DETECTADO!")
            stats.append("   💡 Listo para manualidades creativas")
        elif context.get('potential_kit'):
            stats.append("   🔄 POTENCIAL KIT DETECTADO")
            stats.append("   💡 Botella y tijeras presentes - ¡Perfecto para reciclar!")
        elif context.get('has_bottle'):
            stats.append("   🍼 Solo botella detectada")
            stats.append("   💡 Consigue tijeras para completar el kit")
        elif context.get('has_scissors'):
            stats.append("   ✂️ Solo tijeras detectadas")
            stats.append("   💡 Consigue botella para completar el kit")
        else:
            stats.append("   🔍 Buscando componentes de reciclaje")
            stats.append("   💡 Necesitas: botella + tijeras")
        
        # Objetos individuales
        stats.append("\n🟢 OBJETOS INDIVIDUALES DETECTADOS:")
        if self.results['pretrained']:
            for cls, count in self.results['pretrained'].items():
                emoji = self.emojis.get(cls, '📦')
                if cls == 'bottle':
                    stats.append(f"   {emoji} Botellas reciclables: {count}")
                elif cls == 'scissors':
                    stats.append(f"   {emoji} Tijeras para manualidades: {count}")
                else:
                    stats.append(f"   {emoji} {cls}: {count}")
        else:
            stats.append("   ❌ Sin objetos individuales detectados")
            stats.append("   💡 Prueba con:")
            stats.append("      🍼 Botella de agua/refresco")
            stats.append("      ✂️ Tijeras de cocina/oficina")
        
        # Kit personalizado
        stats.append("\n🔵 KIT DE RECICLAJE (Modelo Personalizado):")
        if self.custom_model is None:
            stats.append("   ⏳ Modelo no entrenado")
            stats.append("   🚀 Para detectar kits de reciclaje:")
            stats.append("      1. Recolecta 50+ fotos de botella+tijeras juntas")
            stats.append("      2. Anótalas como 'kit_reciclaje'")
            stats.append("      3. Entrena el modelo")
        elif self.results['custom']:
            for cls, count in self.results['custom'].items():
                stats.append(f"   ♻️ {cls}: {count} kits")
        else:
            stats.append("   ❌ Sin kits detectados")
            stats.append("   💡 El modelo está entrenado pero no encuentra kits")
        
        # Resumen
        total = sum(self.results['pretrained'].values()) + sum(self.results['custom'].values())
        stats.append(f"\n📊 RESUMEN:")
        stats.append(f"   🎯 Total objetos: {total}")
        stats.append(f"   ⏱️ Tiempo: {self.results.get('time', 0):.2f}s")
        
        # Consejos específicos
        stats.append(f"\n💡 CONSEJOS PARA RECICLAJE CREATIVO:")
        stats.append(f"   🍼 Botellas: plástico transparente funciona mejor")
        stats.append(f"   ✂️ Tijeras: cualquier tipo (cocina, oficina, manualidades)")
        stats.append(f"   ♻️ Kit: ambos objetos visibles en la misma imagen")  
        stats.append(f"   📱 Confianza: 0.2 para detectar más objetos")
        
        return "\n".join(stats)
    
    def generate_video_stats(self, frames, processing_time, potential_kits):
        """Genera estadísticas de video con análisis de reciclaje."""
        stats = []
        stats.append("🎬 ANÁLISIS DE VIDEO - RECICLAJE CREATIVO")
        stats.append("="*60)
        
        fps = frames / processing_time if processing_time > 0 else 0
        kit_percentage = (potential_kits / frames * 100) if frames > 0 else 0
        
        stats.append(f"\n📊 PROCESAMIENTO:")
        stats.append(f"   🎞️ Frames: {frames}")
        stats.append(f"   ⏱️ Tiempo: {processing_time:.2f}s")
        stats.append(f"   🚀 Velocidad: {fps:.1f} FPS")
        
        stats.append(f"\n♻️ ANÁLISIS DE RECICLAJE:")
        stats.append(f"   🔄 Frames con potencial kit: {potential_kits}")
        stats.append(f"   📈 Porcentaje de kit: {kit_percentage:.1f}%")
        
        if kit_percentage > 50:
            stats.append(f"   ✅ Video ideal para reciclaje creativo!")
        elif kit_percentage > 20:
            stats.append(f"   🔄 Buen potencial para manualidades")
        else:
            stats.append(f"   💡 Pocos componentes de reciclaje detectados")
        
        stats.append(f"\n🟢 OBJETOS INDIVIDUALES ACUMULADOS:")
        if self.results['pretrained']:
            total = sum(self.results['pretrained'].values())
            avg = total / frames if frames > 0 else 0
            stats.append(f"   📈 Total: {total} (promedio: {avg:.2f} por frame)")
            for cls, count in self.results['pretrained'].items():
                emoji = self.emojis.get(cls, '📦')
                percentage = (count / total * 100) if total > 0 else 0
                stats.append(f"   {emoji} {cls}: {count} ({percentage:.1f}%)")
        else:
            stats.append("   ❌ Sin objetos detectados en el video")
        
        stats.append(f"\n🔵 KITS DE RECICLAJE:")
        if self.results['custom']:
            for cls, count in self.results['custom'].items():
                stats.append(f"   ♻️ {cls}: {count} kits")
        else:
            stats.append("   ⏳ Sin kits detectados (modelo personalizado)")
        
        return "\n".join(stats)
    
    def export_csv(self):
        """Exporta resultados con contexto de reciclaje."""
        data = []
        timestamp = datetime.now().isoformat()
        
        # Objetos individuales
        for cls, count in self.results['pretrained'].items():
            emoji = self.emojis.get(cls, '📦')
            purpose = "Reciclable" if cls == 'bottle' else "Manualidad" if cls == 'scissors' else "General"
            data.append({
                'modelo': 'preentrenado',
                'tipo': 'objeto_individual',
                'clase': cls,
                'proposito': purpose,
                'emoji': emoji,
                'cantidad': count,
                'timestamp': timestamp
            })
        
        # Kits de reciclaje
        for cls, count in self.results['custom'].items():
            data.append({
                'modelo': 'personalizado',
                'tipo': 'kit_completo',
                'clase': cls,
                'proposito': 'Reciclaje_Creativo',
                'emoji': '♻️',
                'cantidad': count,
                'timestamp': timestamp
            })
        
        # Análisis contextual
        context = self.results.get('context', {})
        if context:
            data.append({
                'modelo': 'analisis',
                'tipo': 'contexto',
                'clase': 'potencial_kit',
                'proposito': 'Análisis',
                'emoji': '🔄',
                'cantidad': 1 if context.get('potential_kit') else 0,
                'timestamp': timestamp
            })
        
        if data:
            df = pd.DataFrame(data)
            filename = f"reciclaje_detecciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            return filename
        return "Sin datos"

# Crear detector global
detector = DetectorReciclaje()

def process_image_gradio(image, confidence):
    """Procesar imagen en Gradio."""
    return detector.process_image(image, confidence)

def process_video_gradio(video, confidence, max_frames):
    """Procesar video en Gradio.""" 
    return detector.process_video(video, confidence, max_frames)

def export_csv_gradio():
    """Exportar CSV en Gradio."""
    filename = detector.export_csv()
    if filename != "Sin datos":
        return f"✅ Exportado: {filename}"
    return "❌ Sin datos para exportar"

def create_demo_video():
    """Crear video de demostración compatible con Gradio - VERSION CORREGIDA."""
    try:
        print("🎬 Creando video demo compatible...")
        
        width, height = 640, 480
        fps = 15  # FPS más bajo para mejor compatibilidad
        duration = 8  # 8 segundos
        
        os.makedirs("demo_videos", exist_ok=True)
        
        # Nombre único para evitar conflictos
        timestamp = int(time.time())
        output_path = f"demo_videos/demo_reciclaje_{timestamp}.avi"  # Cambio a .avi
        
        # Codec más compatible
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Verificar que el writer funciona
        if not out.isOpened():
            print("⚠️ Intentando codec alternativo...")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            output_path = f"demo_videos/demo_reciclaje_{timestamp}.avi"
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            return "❌ No se puede crear video demo"
        
        total_frames = fps * duration
        print(f"📊 Generando {total_frames} frames...")
        
        for frame_num in range(total_frames):
            # Crear frame base más colorido
            frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Fondo gris claro
            
            # Título más visible
            cv2.rectangle(frame, (0, 0), (width, 60), (50, 50, 50), -1)
            cv2.putText(frame, "DEMO: RECICLAJE CREATIVO", (120, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Animación de tiempo
            progress = frame_num / total_frames
            
            # Botella animada (rectángulo azul que se mueve)
            bottle_x = 80 + int(60 * np.sin(frame_num * 0.2))
            bottle_y = 120
            bottle_w, bottle_h = 60, 140
            
            # Dibujar botella con mejor forma
            cv2.rectangle(frame, (bottle_x, bottle_y), (bottle_x + bottle_w, bottle_y + bottle_h), 
                         (255, 150, 100), -1)  # Azul claro
            cv2.rectangle(frame, (bottle_x, bottle_y), (bottle_x + bottle_w, bottle_y + bottle_h), 
                         (200, 100, 50), 3)   # Borde
            
            # Etiqueta de botella
            cv2.rectangle(frame, (bottle_x - 5, bottle_y - 35), (bottle_x + 70, bottle_y - 5), 
                         (0, 255, 0), -1)
            cv2.putText(frame, "BOTELLA", (bottle_x, bottle_y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Tijeras animadas (dos rectángulos que se mueven)
            scissors_x = 350 + int(40 * np.cos(frame_num * 0.25))
            scissors_y = 150
            scissors_w, scissors_h = 100, 20
            
            # Dibujar tijeras (dos hojas)
            cv2.rectangle(frame, (scissors_x, scissors_y), 
                         (scissors_x + scissors_w, scissors_y + scissors_h), 
                         (100, 255, 100), -1)  # Verde claro
            cv2.rectangle(frame, (scissors_x, scissors_y + 30), 
                         (scissors_x + scissors_w, scissors_y + scissors_h + 30), 
                         (100, 255, 100), -1)  # Segunda hoja
            
            # Bordes de tijeras
            cv2.rectangle(frame, (scissors_x, scissors_y), 
                         (scissors_x + scissors_w, scissors_y + scissors_h), 
                         (50, 200, 50), 3)
            cv2.rectangle(frame, (scissors_x, scissors_y + 30), 
                         (scissors_x + scissors_w, scissors_y + scissors_h + 30), 
                         (50, 200, 50), 3)
            
            # Etiqueta de tijeras
            cv2.rectangle(frame, (scissors_x - 5, scissors_y - 35), (scissors_x + 85, scissors_y - 5), 
                         (0, 255, 0), -1)
            cv2.putText(frame, "TIJERAS", (scissors_x, scissors_y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # En la segunda mitad, mostrar kit de reciclaje
            if progress > 0.4:  # Después del 40%
                kit_x = 200
                kit_y = 300
                
                # Marco del kit (rectángulo de color distintivo)
                cv2.rectangle(frame, (kit_x - 30, kit_y - 30), (kit_x + 200, kit_y + 100), 
                             (255, 0, 255), 4)  # Marco magenta
                
                # Fondo del kit
                cv2.rectangle(frame, (kit_x - 25, kit_y - 25), (kit_x + 195, kit_y + 95), 
                             (255, 255, 255), -1)  # Fondo blanco
                
                # Botella pequeña en el kit
                cv2.rectangle(frame, (kit_x, kit_y), (kit_x + 40, kit_y + 70), 
                             (255, 150, 100), -1)
                cv2.rectangle(frame, (kit_x, kit_y), (kit_x + 40, kit_y + 70), 
                             (200, 100, 50), 2)
                
                # Tijeras pequeñas en el kit
                cv2.rectangle(frame, (kit_x + 60, kit_y + 20), (kit_x + 130, kit_y + 35), 
                             (100, 255, 100), -1)
                cv2.rectangle(frame, (kit_x + 60, kit_y + 40), (kit_x + 130, kit_y + 55), 
                             (100, 255, 100), -1)
                
                # Etiqueta del kit
                cv2.rectangle(frame, (kit_x - 25, kit_y - 55), (kit_x + 140, kit_y - 30), 
                             (255, 0, 0), -1)  # Fondo azul
                cv2.putText(frame, "KIT RECICLAJE", (kit_x - 15, kit_y - 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Información del frame (más visible)
            info_y = height - 40
            cv2.rectangle(frame, (0, info_y), (width, height), (0, 0, 0), -1)
            
            frame_info = f"Frame {frame_num + 1}/{total_frames} | Progreso: {progress*100:.0f}%"
            cv2.putText(frame, frame_info, (10, info_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Escribir frame con validación
            if frame.shape == (height, width, 3):
                out.write(frame)
            else:
                print(f"⚠️ Frame {frame_num} con dimensiones incorrectas")
                frame_resized = cv2.resize(frame, (width, height))
                out.write(frame_resized)
            
            # Progreso cada 25 frames
            if (frame_num + 1) % 25 == 0:
                print(f"📊 Generando: {((frame_num + 1)/total_frames)*100:.0f}%")
        
        out.release()
        
        # Verificar que el archivo se creó correctamente
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Video demo creado: {output_path}")
            print(f"📊 Tamaño: {file_size/1024/1024:.1f} MB")
            
            if file_size > 10000:  # Al menos 10KB
                return f"✅ Video demo creado exitosamente: {output_path}\n📊 Tamaño: {file_size/1024/1024:.1f} MB\n🎬 Listo para subir a la aplicación"
            else:
                return f"⚠️ Video creado pero muy pequeño. Revisa el codec."
        else:
            return "❌ No se pudo crear el archivo de video"
        
    except Exception as e:
        print(f"❌ Error creando video demo: {e}")
        return f"❌ Error creando video demo: {str(e)}. Intenta instalaciones: pip install opencv-python"

def create_interface():
    """Crear interfaz Gradio con tema de reciclaje."""
    
    with gr.Blocks(title="♻️ Detector de Reciclaje Creativo - UPChiapas", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # ♻️ Detector de Reciclaje Creativo con YOLO
        ## Universidad Politécnica de Chiapas | Proyecto Coherente
        
        ### 🎯 ESTRATEGIA COHERENTE:
        
        #### 🟢 MODELO PREENTRENADO (Objetos Individuales):
        - 🍼 **BOTELLAS** → Materiales reciclables (plástico, vidrio)
        - ✂️ **TIJERAS** → Herramientas para manualidades
        
        #### 🔵 MODELO PERSONALIZADO (Concepto Integrado):
        - ♻️ **KIT DE RECICLAJE** → Botella + tijeras juntas para manualidades
        
        ### 💡 CONCEPTO DEL PROYECTO:
        **"Reciclaje Creativo"** - Detectamos los componentes individuales (botella, tijeras) y 
        reconocemos cuando están juntos formando un kit para hacer manualidades reciclando plástico.
        
        ### 📋 Instrucciones:
        1. **Objetos individuales**: Sube fotos con botellas O tijeras
        2. **Kit completo**: Sube fotos con botella Y tijeras juntas  
        3. **Confianza baja (0.2)**: Para detectar más objetos
        4. **Videos**: Observa el análisis frame por frame
        """)
        
        with gr.Tabs():
            # IMÁGENES
            with gr.Tab("📸 Detección en Imágenes"):
                gr.Markdown("### Analiza imágenes buscando componentes de reciclaje")
                
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(
                            label="📸 Subir Imagen (botellas, tijeras, o ambos)",
                            type="numpy"
                        )
                        
                        confidence_img = gr.Slider(
                            0.1, 1.0, 0.25, step=0.05,
                            label="🎚️ Confianza (0.2 recomendado)"
                        )
                        
                        process_img_btn = gr.Button("♻️ ANALIZAR RECICLAJE", variant="primary")
                        
                        gr.Markdown("""
                        ### 💡 Objetos que Funcioann:
                        - 🍼 **Botellas**: Agua, refresco, cualquier plástica
                        - ✂️ **Tijeras**: Cocina, oficina, manualidades
                        - ♻️ **Kit**: Ambos objetos en la misma imagen
                        
                        ### 🎯 Colores:
                        - **🟢 Verde**: Objetos individuales
                        - **🔵 Azul**: Kit de reciclaje completo
                        """)
                    
                    with gr.Column():
                        output_image = gr.Image(label="♻️ ANÁLISIS DE RECICLAJE")
                        stats_img = gr.Textbox(
                            label="📊 Análisis Contextual",
                            lines=15,
                            interactive=False
                        )
            
            # VIDEOS
            with gr.Tab("🎬 Análisis de Videos"):
                gr.Markdown("### Analiza videos completos buscando patrones de reciclaje")
                
                with gr.Row():
                    with gr.Column():
                        input_video = gr.Video(label="🎬 Subir Video")
                        
                        confidence_vid = gr.Slider(
                            0.1, 1.0, 0.25, step=0.05,
                            label="🎚️ Confianza"
                        )
                        
                        max_frames = gr.Slider(
                            20, 200, 60, step=10,
                            label="🎞️ Frames máximos"
                        )
                        
                        process_vid_btn = gr.Button("🎬 PROCESAR VIDEO", variant="primary")
                        
                        # Botón para crear demo
                        create_demo_btn = gr.Button("🎨 Crear Video Demo", variant="secondary")
                        demo_status = gr.Textbox(label="Estado Demo")
                    
                    with gr.Column():
                        output_video = gr.Video(label="♻️ VIDEO CON ANÁLISIS")
                        stats_vid = gr.Textbox(
                            label="📊 Análisis Completo del Video",
                            lines=15,
                            interactive=False
                        )
        
        # EXPORTACIÓN
        with gr.Row():
            export_btn = gr.Button("📊 Exportar Análisis a CSV", variant="secondary")
            export_status = gr.Textbox(label="Estado Exportación")
        
        # JSON
        with gr.Accordion("🔧 Datos Técnicos JSON", open=False):
            json_output = gr.JSON(label="Datos Completos")
        
        gr.Markdown("""
        ---
        ### 🎯 Lógica del Proyecto:
        
        #### 📝 **Coherencia Temática**:
        - **Tema**: Reciclaje creativo y manualidades
        - **Objetos individuales**: Botellas (material) + Tijeras (herramienta)  
        - **Concepto integrado**: Kit de reciclaje (ambos juntos)
        
        #### 🔬 **Aspectos Técnicos**:
        - **Modelo preentrenado**: Detecta objetos del dataset COCO
        - **Modelo personalizado**: Entrenado para reconocer kits específicos
        - **Visualización**: Colores diferenciados y análisis contextual
        
        #### 📊 **Estado del Proyecto**:
        - ✅ **75% COMPLETO**: Detecta botellas y tijeras perfectamente
        - ⏳ **25% PENDIENTE**: Entrenar modelo para kits de reciclaje
        
        #### 🎓 **Para la Evaluación**:
        - **Demuestra**: Detección funcional inmediata
        - **Explica**: La relación lógica entre los objetos
        - **Documenta**: Proceso de entrenamiento del modelo personalizado
        
        ---
        **👥 Desarrollado por**: [Tu Nombre] y [Compañero]  
        **🏫 Universidad**: Politécnica de Chiapas  
        **📚 Materia**: Multimedia y Diseño Digital
        """)
        
        # CONECTAR EVENTOS
        process_img_btn.click(
            fn=process_image_gradio,
            inputs=[input_image, confidence_img],
            outputs=[output_image, stats_img, json_output]
        )
        
        process_vid_btn.click(
            fn=process_video_gradio,
            inputs=[input_video, confidence_vid, max_frames],
            outputs=[output_video, stats_vid, json_output]
        )
        
        export_btn.click(
            fn=export_csv_gradio,
            outputs=export_status
        )
        
        create_demo_btn.click(
            fn=create_demo_video,
            outputs=demo_status
        )
    
    return interface

if __name__ == "__main__":
    print("♻️ DETECTOR DE RECICLAJE CREATIVO")
    print("="*60)
    print("🏫 Universidad Politécnica de Chiapas")
    print("📚 Proyecto: Detección Coherente con YOLO")
    print()
    print("🎯 ESTRATEGIA COHERENTE:")
    print("   🟢 INDIVIDUALES: 🍼 Botellas + ✂️ Tijeras")
    print("   🔵 INTEGRADO: ♻️ Kit de Reciclaje")
    print()
    print("💡 CONCEPTO: Reciclaje Creativo")
    print("   - Detecta materiales (botellas)")
    print("   - Detecta herramientas (tijeras)")  
    print("   - Reconoce kits completos (ambos juntos)")
    print()
    print("🌐 Interfaz: http://localhost:7860")
    print("🎬 Videos: Con análisis contextual frame por frame")
    print()
    
    try:
        app = create_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Instala: pip install ultralytics opencv-python gradio")