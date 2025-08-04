#!/usr/bin/env python3
"""
Genera un video de prueba para demostrar la funcionalidad
Universidad Politécnica de Chiapas

Este script crea un video de ejemplo con objetos simulados
para probar la detección de útiles escolares.
"""

import cv2
import numpy as np
import os

def create_test_video():
    """Crea un video de prueba con objetos simulados."""
    print("🎬 Creando video de prueba...")
    
    # Configuración del video
    width, height = 640, 480
    fps = 30
    duration = 10  # segundos
    total_frames = fps * duration
    
    # Crear carpeta si no existe
    os.makedirs("test_videos", exist_ok=True)
    output_path = "test_videos/demo_utiles_escolares.mp4"
    
    # Configurar writer de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📊 Generando {total_frames} frames...")
    
    for frame_num in range(total_frames):
        # Crear frame base (fondo blanco)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Tiempo actual en el video
        time_progress = frame_num / total_frames
        
        # Simular escritorio con objetos
        draw_desktop_scene(frame, time_progress, frame_num)
        
        # Agregar información del frame
        cv2.putText(frame, f"Frame: {frame_num + 1}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.putText(frame, "Video de Prueba - Utiles Escolares", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        
        # Escribir frame
        out.write(frame)
        
        # Mostrar progreso
        if frame_num % 60 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"📊 Progreso: {progress:.1f}%")
    
    out.release()
    print(f"✅ Video creado: {output_path}")
    return output_path

def draw_desktop_scene(frame, time_progress, frame_num):
    """Dibuja una escena de escritorio con útiles escolares simulados."""
    height, width = frame.shape[:2]
    
    # Simular mesa (rectángulo marrón en la parte inferior)
    table_color = (139, 69, 19)  # Marrón
    cv2.rectangle(frame, (0, int(height * 0.7)), (width, height), table_color, -1)
    
    # Animación: objetos que se mueven y aparecen
    phase = time_progress * 2 * np.pi
    
    # Simular cuaderno (rectángulo azul)
    notebook_x = int(width * 0.3 + 50 * np.sin(phase * 0.5))
    notebook_y = int(height * 0.8)
    notebook_w, notebook_h = 120, 80
    cv2.rectangle(frame, 
                 (notebook_x, notebook_y), 
                 (notebook_x + notebook_w, notebook_y + notebook_h),
                 (255, 100, 100), -1)  # Azul claro
    
    # Bordes del cuaderno
    cv2.rectangle(frame, 
                 (notebook_x, notebook_y), 
                 (notebook_x + notebook_w, notebook_y + notebook_h),
                 (200, 50, 50), 3)
    
    # Texto en el cuaderno
    cv2.putText(frame, "Cuaderno", 
               (notebook_x + 10, notebook_y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Simular lápiz (línea amarilla con punta)
    pencil_x = int(width * 0.6 + 30 * np.cos(phase))
    pencil_y = int(height * 0.75)
    pencil_end_x = pencil_x + 100
    pencil_end_y = pencil_y - 10
    
    # Cuerpo del lápiz
    cv2.line(frame, (pencil_x, pencil_y), (pencil_end_x, pencil_end_y), (0, 255, 255), 8)
    
    # Punta del lápiz
    cv2.circle(frame, (pencil_end_x, pencil_end_y), 4, (0, 0, 0), -1)
    
    # Texto del lápiz
    cv2.putText(frame, "Lapiz", 
               (pencil_x, pencil_y - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Simular kit de escritura (cuando aparecen juntos)
    if time_progress > 0.5:  # Aparece en la segunda mitad
        kit_x = int(width * 0.1)
        kit_y = int(height * 0.85)
        
        # Marco del kit
        cv2.rectangle(frame, (kit_x - 10, kit_y - 20), (kit_x + 140, kit_y + 60), (0, 255, 0), 3)
        
        # Cuaderno pequeño
        cv2.rectangle(frame, (kit_x, kit_y), (kit_x + 60, kit_y + 40), (255, 100, 100), -1)
        
        # Lápiz pequeño
        cv2.line(frame, (kit_x + 70, kit_y + 10), (kit_x + 120, kit_y + 5), (0, 255, 255), 4)
        
        # Etiqueta del kit
        cv2.putText(frame, "Kit Escritura", 
                   (kit_x, kit_y - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 2)
    
    # Simular persona (círculo para cabeza y rectángulo para cuerpo)
    if frame_num % 120 < 60:  # Aparece intermitentemente
        person_x = int(width * 0.8)
        person_y = int(height * 0.4)
        
        # Cabeza
        cv2.circle(frame, (person_x, person_y), 30, (200, 180, 160), -1)
        
        # Cuerpo
        cv2.rectangle(frame, (person_x - 25, person_y + 20), (person_x + 25, person_y + 100), (100, 100, 200), -1)
        
        # Etiqueta
        cv2.putText(frame, "Estudiante", 
                   (person_x - 40, person_y - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Simular libro (rectángulo grueso)
    book_x = int(width * 0.15)
    book_y = int(height * 0.6 + 20 * np.sin(phase * 2))
    cv2.rectangle(frame, (book_x, book_y), (book_x + 80, book_y + 120), (0, 100, 200), -1)
    cv2.rectangle(frame, (book_x, book_y), (book_x + 80, book_y + 120), (0, 50, 150), 3)
    
    cv2.putText(frame, "Libro", 
               (book_x + 10, book_y + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

def main():
    """Función principal."""
    print("🎬 GENERADOR DE VIDEO DE PRUEBA")
    print("="*50)
    print("Este script crea un video de ejemplo para probar")
    print("la funcionalidad de detección de útiles escolares.")
    print()
    
    try:
        video_path = create_test_video()
        
        print("\n✅ ¡Video de prueba generado exitosamente!")
        print(f"📁 Ubicación: {video_path}")
        print("\n💡 Cómo usar:")
        print("1. Ejecuta: python app.py")
        print("2. Ve a la pestaña 'Procesamiento de Videos'")
        print("3. Carga el video generado")
        print("4. ¡Observa las detecciones!")
        
    except Exception as e:
        print(f"❌ Error generando video: {e}")

if __name__ == "__main__":
    main()