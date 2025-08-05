"""
Analizador personalizado para Botellas y Tijeras
Integración directa con el sistema YOLO Detector
"""

import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import math

class BottleScissorsAnalyzer:
    """
    Clase personalizada para análisis específico de botellas y tijeras
    Funcionalidades:
    - Conteo y clasificación por tamaño
    - Análisis de distribución espacial
    - Detección de agrupaciones
    - Generación de reportes automáticos
    - Alertas de seguridad (tijeras cerca de otras personas)
    """
    
    def __init__(self):
        print("🔬 Inicializando BottleScissorsAnalyzer...")
        
        # Contadores y estadísticas
        self.reset_statistics()
        
        # Configuración de análisis
        self.size_thresholds = {
            'bottle_small': 1500,    # píxeles²
            'bottle_medium': 4000,
            'bottle_large': 8000,
            'scissors_small': 800,
            'scissors_medium': 2000,
            'scissors_large': 4000
        }
        
        # Colores para visualización
        self.colors = {
            'bottle_small': (0, 255, 128),     # Verde claro
            'bottle_medium': (0, 200, 255),    # Azul claro  
            'bottle_large': (0, 100, 255),     # Azul fuerte
            'scissors_small': (255, 128, 0),   # Naranja claro
            'scissors_medium': (255, 100, 0),  # Naranja
            'scissors_large': (255, 50, 0),    # Rojo-naranja
            'danger_zone': (0, 0, 255),        # Rojo para alertas
            'cluster': (255, 255, 0)           # Amarillo para grupos
        }
        
        print("✅ BottleScissorsAnalyzer listo")
    
    def reset_statistics(self):
        """Reiniciar estadísticas"""
        self.statistics = {
            'bottles': {
                'total': 0,
                'small': 0,
                'medium': 0,
                'large': 0,
                'positions': [],
                'sizes': []
            },
            'scissors': {
                'total': 0,
                'small': 0,
                'medium': 0,
                'large': 0,
                'positions': [],
                'sizes': []
            },
            'clusters': [],
            'danger_zones': [],
            'analysis_timestamp': None
        }
    
    def analyze_detections(self, image, detections, person_detections=None):
        """
        Análisis principal de botellas y tijeras
        
        Args:
            image: Imagen original
            detections: Lista de detecciones con formato:
                       [{'clase': str, 'confianza': float, 'bbox': [x1,y1,x2,y2]}]
            person_detections: Lista opcional de detecciones de personas para análisis de seguridad
        
        Returns:
            tuple: (imagen_anotada, estadísticas_detalladas)
        """
        print("🔍 Analizando botellas y tijeras...")
        
        # Reiniciar estadísticas para este análisis
        self.reset_statistics()
        self.statistics['analysis_timestamp'] = datetime.now().isoformat()
        
        # Imagen para análisis (copia)
        analysis_image = image.copy()
        height, width = analysis_image.shape[:2]
        
        # Filtrar detecciones relevantes
        bottles = [d for d in detections if 'botella' in d['clase'].lower()]
        scissors = [d for d in detections if 'tijera' in d['clase'].lower()]
        
        # Analizar botellas
        if bottles:
            self._analyze_bottles(analysis_image, bottles, width, height)
        
        # Analizar tijeras
        if scissors:
            self._analyze_scissors(analysis_image, scissors, width, height)
        
        # Detectar agrupaciones
        if bottles or scissors:
            self._detect_clusters(analysis_image, bottles + scissors)
        
        # Análisis de seguridad (tijeras cerca de personas)
        if scissors and person_detections:
            self._analyze_safety(analysis_image, scissors, person_detections)
        
        # Dibujar panel de estadísticas
        self._draw_analysis_panel(analysis_image, width, height)
        
        print(f"✅ Análisis completado: {len(bottles)} botellas, {len(scissors)} tijeras")
        
        return analysis_image, self.statistics
    
    def _analyze_bottles(self, image, bottles, width, height):
        """Análisis específico de botellas"""
        for bottle in bottles:
            bbox = bottle['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calcular tamaño
            area = (x2 - x1) * (y2 - y1)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Clasificar por tamaño
            if area < self.size_thresholds['bottle_small']:
                size_category = 'small'
                color = self.colors['bottle_small']
                label = "BOTELLA-S"
            elif area < self.size_thresholds['bottle_medium']:
                size_category = 'medium'
                color = self.colors['bottle_medium']
                label = "BOTELLA-M"
            else:
                size_category = 'large'
                color = self.colors['bottle_large']
                label = "BOTELLA-L"
            
            # Actualizar estadísticas
            self.statistics['bottles']['total'] += 1
            self.statistics['bottles'][size_category] += 1
            self.statistics['bottles']['positions'].append(center)
            self.statistics['bottles']['sizes'].append(area)
            
            # Dibujar análisis visual
            self._draw_enhanced_detection(image, bbox, label, 
                                        bottle['confianza'], color, area)
    
    def _analyze_scissors(self, image, scissors, width, height):
        """Análisis específico de tijeras"""
        for scissor in scissors:
            bbox = scissor['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calcular tamaño
            area = (x2 - x1) * (y2 - y1)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Clasificar por tamaño
            if area < self.size_thresholds['scissors_small']:
                size_category = 'small'
                color = self.colors['scissors_small']
                label = "TIJERAS-S"
            elif area < self.size_thresholds['scissors_medium']:
                size_category = 'medium'
                color = self.colors['scissors_medium']
                label = "TIJERAS-M"
            else:
                size_category = 'large'
                color = self.colors['scissors_large']
                label = "TIJERAS-L"
            
            # Actualizar estadísticas
            self.statistics['scissors']['total'] += 1
            self.statistics['scissors'][size_category] += 1
            self.statistics['scissors']['positions'].append(center)
            self.statistics['scissors']['sizes'].append(area)
            
            # Dibujar análisis visual
            self._draw_enhanced_detection(image, bbox, label, 
                                        scissor['confianza'], color, area)
    
    def _detect_clusters(self, image, all_objects):
        """Detectar agrupaciones de objetos"""
        if len(all_objects) < 2:
            return
        
        positions = []
        for obj in all_objects:
            bbox = obj['bbox']
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            positions.append(center)
        
        # Detectar clusters usando distancia simple
        clusters = []
        cluster_distance = 150  # píxeles
        
        for i, pos1 in enumerate(positions):
            cluster = [i]
            for j, pos2 in enumerate(positions[i+1:], i+1):
                distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                if distance < cluster_distance:
                    cluster.append(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        # Dibujar clusters
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) > 1:
                cluster_positions = [positions[i] for i in cluster]
                self._draw_cluster(image, cluster_positions, cluster_idx)
                self.statistics['clusters'].append({
                    'objects_count': len(cluster),
                    'center': self._calculate_cluster_center(cluster_positions)
                })
    
    def _analyze_safety(self, image, scissors, persons):
        """Análisis de seguridad: tijeras cerca de personas"""
        danger_distance = 100  # píxeles
        
        for scissor in scissors:
            s_bbox = scissor['bbox']
            s_center = ((s_bbox[0] + s_bbox[2]) // 2, (s_bbox[1] + s_bbox[3]) // 2)
            
            for person in persons:
                p_bbox = person['bbox']
                p_center = ((p_bbox[0] + p_bbox[2]) // 2, (p_bbox[1] + p_bbox[3]) // 2)
                
                distance = math.sqrt((s_center[0] - p_center[0])**2 + 
                                   (s_center[1] - p_center[1])**2)
                
                if distance < danger_distance:
                    # Marcar zona de peligro
                    self._draw_danger_zone(image, s_center, p_center)
                    self.statistics['danger_zones'].append({
                        'scissors_position': s_center,
                        'person_position': p_center,
                        'distance': distance
                    })
    
    def _draw_enhanced_detection(self, image, bbox, label, confidence, color, area):
        """Dibujar detección mejorada con información de tamaño"""
        x1, y1, x2, y2 = bbox
        
        # Marco principal
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Información detallada
        info_text = f"{label}"
        size_text = f"Area: {area}px²"
        conf_text = f"Conf: {confidence:.2f}"
        
        # Fondo para texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        
        # Calcular tamaño del texto
        (text_w1, text_h1), _ = cv2.getTextSize(info_text, font, font_scale, 1)
        (text_w2, text_h2), _ = cv2.getTextSize(size_text, font, font_scale, 1)
        (text_w3, text_h3), _ = cv2.getTextSize(conf_text, font, font_scale, 1)
        
        max_width = max(text_w1, text_w2, text_w3)
        total_height = text_h1 + text_h2 + text_h3 + 12
        
        # Fondo del panel de información
        cv2.rectangle(image, (x1, y1 - total_height - 5), 
                     (x1 + max_width + 10, y1), (0, 0, 0), -1)
        cv2.rectangle(image, (x1, y1 - total_height - 5), 
                     (x1 + max_width + 10, y1), color, 1)
        
        # Textos
        cv2.putText(image, info_text, (x1 + 5, y1 - total_height + text_h1), 
                   font, font_scale, color, 1)
        cv2.putText(image, size_text, (x1 + 5, y1 - total_height + text_h1 + text_h2 + 4), 
                   font, font_scale, (255, 255, 255), 1)
        cv2.putText(image, conf_text, (x1 + 5, y1 - total_height + text_h1 + text_h2 + text_h3 + 8), 
                   font, font_scale, (255, 255, 255), 1)
    
    def _draw_cluster(self, image, positions, cluster_id):
        """Dibujar agrupación de objetos"""
        if len(positions) < 2:
            return
        
        # Calcular centro del cluster
        center = self._calculate_cluster_center(positions)
        
        # Dibujar círculo de cluster
        max_distance = max([math.sqrt((pos[0] - center[0])**2 + (pos[1] - center[1])**2) 
                           for pos in positions])
        
        cv2.circle(image, center, int(max_distance + 20), self.colors['cluster'], 2)
        
        # Etiqueta de cluster
        label = f"GRUPO-{cluster_id+1} ({len(positions)} objetos)"
        cv2.putText(image, label, (center[0] - 50, center[1] - int(max_distance) - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['cluster'], 2)
    
    def _draw_danger_zone(self, image, scissors_pos, person_pos):
        """Dibujar zona de peligro"""
        # Línea de conexión
        cv2.line(image, scissors_pos, person_pos, self.colors['danger_zone'], 3)
        
        # Alerta visual
        cv2.circle(image, scissors_pos, 30, self.colors['danger_zone'], 3)
        cv2.putText(image, "PELIGRO!", (scissors_pos[0] - 30, scissors_pos[1] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['danger_zone'], 2)
    
    def _calculate_cluster_center(self, positions):
        """Calcular centro de un cluster"""
        if not positions:
            return (0, 0)
        
        center_x = sum(pos[0] for pos in positions) // len(positions)
        center_y = sum(pos[1] for pos in positions) // len(positions)
        return (center_x, center_y)
    
    def _draw_analysis_panel(self, image, width, height):
        """Dibujar panel de análisis en la imagen"""
        panel_height = 100
        panel_y = height - panel_height
        
        # Fondo del panel
        overlay = image.copy()
        cv2.rectangle(overlay, (0, panel_y), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        # Línea divisoria
        cv2.line(image, (0, panel_y), (width, panel_y), (0, 255, 255), 2)
        
        # Título del panel
        cv2.putText(image, "ANALISIS PERSONALIZADO - BOTELLAS Y TIJERAS", 
                   (10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Estadísticas de botellas
        bottle_stats = self.statistics['bottles']
        bottle_text = f"BOTELLAS: {bottle_stats['total']} (S:{bottle_stats['small']} M:{bottle_stats['medium']} L:{bottle_stats['large']})"
        cv2.putText(image, bottle_text, (10, panel_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)
        
        # Estadísticas de tijeras
        scissors_stats = self.statistics['scissors']
        scissors_text = f"TIJERAS: {scissors_stats['total']} (S:{scissors_stats['small']} M:{scissors_stats['medium']} L:{scissors_stats['large']})"
        cv2.putText(image, scissors_text, (10, panel_y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
        
        # Información adicional
        clusters_count = len(self.statistics['clusters'])
        dangers_count = len(self.statistics['danger_zones'])
        
        additional_text = f"GRUPOS: {clusters_count} | ALERTAS: {dangers_count}"
        cv2.putText(image, additional_text, (width - 300, panel_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def get_detailed_report(self):
        """Generar reporte detallado en formato diccionario"""
        if not self.statistics['analysis_timestamp']:
            return None
        
        # Calcular estadísticas adicionales
        total_objects = (self.statistics['bottles']['total'] + 
                        self.statistics['scissors']['total'])
        
        bottle_avg_size = (np.mean(self.statistics['bottles']['sizes']) 
                          if self.statistics['bottles']['sizes'] else 0)
        scissors_avg_size = (np.mean(self.statistics['scissors']['sizes']) 
                            if self.statistics['scissors']['sizes'] else 0)
        
        report = {
            'timestamp': self.statistics['analysis_timestamp'],
            'resumen': {
                'total_objetos': total_objects,
                'total_botellas': self.statistics['bottles']['total'],
                'total_tijeras': self.statistics['scissors']['total'],
                'grupos_detectados': len(self.statistics['clusters']),
                'alertas_seguridad': len(self.statistics['danger_zones'])
            },
            'botellas_detalle': {
                'distribucion_tamaños': {
                    'pequeñas': self.statistics['bottles']['small'],
                    'medianas': self.statistics['bottles']['medium'],
                    'grandes': self.statistics['bottles']['large']
                },
                'tamaño_promedio_px': round(bottle_avg_size, 2),
                'posiciones': self.statistics['bottles']['positions']
            },
            'tijeras_detalle': {
                'distribucion_tamaños': {
                    'pequeñas': self.statistics['scissors']['small'],
                    'medianas': self.statistics['scissors']['medium'],
                    'grandes': self.statistics['scissors']['large']
                },
                'tamaño_promedio_px': round(scissors_avg_size, 2),
                'posiciones': self.statistics['scissors']['positions']
            },
            'analisis_espacial': {
                'clusters': self.statistics['clusters'],
                'zonas_peligro': self.statistics['danger_zones']
            }
        }
        
        return report
    
    def export_report_json(self, filename=None):
        """Exportar reporte a archivo JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analisis_botellas_tijeras_{timestamp}.json"
        
        report = self.get_detailed_report()
        if report:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 Reporte exportado: {filename}")
            return filename
        else:
            print("❌ No hay datos para exportar")
            return None

# Función de integración para usar con el sistema principal
def integrate_with_main_detector(detector_core):
    """
    Función para integrar el analizador con el detector principal
    
    Args:
        detector_core: Instancia de YOLODetectorCore
    
    Returns:
        Función de detección mejorada
    """
    analyzer = BottleScissorsAnalyzer()
    
    def enhanced_detect_objects(image, confidence=0.5, use_custom=True):
        """Detección mejorada con análisis personalizado"""
        # Detección original
        result_image, results = detector_core.detect_objects(image, confidence, use_custom)
        
        # Filtrar detecciones relevantes para análisis
        relevant_detections = [d for d in results.get('detecciones', []) 
                             if any(keyword in d['clase'].lower() 
                                   for keyword in ['botella', 'tijera'])]
        
        # Análisis personalizado si hay objetos relevantes
        if relevant_detections:
            # Buscar detecciones de personas para análisis de seguridad
            person_detections = [d for d in results.get('detecciones', []) 
                               if 'persona' in d['clase'].lower()]
            
            # Realizar análisis personalizado
            enhanced_image, analysis_stats = analyzer.analyze_detections(
                result_image, relevant_detections, person_detections)
            
            # Agregar estadísticas de análisis a los resultados
            results['analisis_personalizado'] = analysis_stats
            results['analyzer_report'] = analyzer.get_detailed_report()
            
            return enhanced_image, results
        
        return result_image, results
    
    return enhanced_detect_objects, analyzer

# Ejemplo de uso independiente
if __name__ == "__main__":
    # Crear analizador
    analyzer = BottleScissorsAnalyzer()
    
    # Ejemplo de detecciones simuladas
    fake_detections = [
        {'clase': 'botella', 'confianza': 0.85, 'bbox': [100, 100, 150, 200]},
        {'clase': 'tijera', 'confianza': 0.92, 'bbox': [300, 150, 350, 200]},
        {'clase': 'botella', 'confianza': 0.78, 'bbox': [500, 300, 600, 500]},
    ]
    
    # Simular imagen
    fake_image = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Realizar análisis
    result_img, stats = analyzer.analyze_detections(fake_image, fake_detections)
    
    # Generar reporte
    report = analyzer.get_detailed_report()
    print("📊 Reporte generado:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # Exportar
    analyzer.export_report_json()
    
    print("✅ Ejemplo de análisis completado")