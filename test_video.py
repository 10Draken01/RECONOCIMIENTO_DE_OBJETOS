#!/usr/bin/env python3
"""
Script para probar y arreglar problemas de video
Universidad Politécnica de Chiapas

Este script:
1. Prueba los codecs disponibles
2. Crea un video de prueba simple
3. Verifica que se pueda leer
4. Da recomendaciones
"""

import cv2
import numpy as np
import os
import tempfile

def test_codecs():
    """Prueba qué codecs están disponibles."""
    print("🧪 PROBANDO CODECS DISPONIBLES...")
    print("-" * 40)
    
    codecs_to_test = [
        ('XVID', 'XVID'),
        ('mp4v', 'MP4V'), 
        ('MJPG', 'MJPG'),
        ('X264', 'X264'),
        ('avc1', 'AVC1')
    ]
    
    width, height = 640, 480
    fps = 15
    
    working_codecs = []
    
    for codec_name, codec_code in codecs_to_test:
        try:
            test_path = f"test_codec_{codec_name}.avi"
            fourcc = cv2.VideoWriter_fourcc(*codec_code)
            out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
            
            if out.isOpened():
                # Escribir un frame de prueba
                test_frame = np.ones((height, width, 3), dtype=np.uint8) * 128
                cv2.putText(test_frame, f"Test {codec_name}", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                out.write(test_frame)
                out.release()
                
                # Verificar que el archivo se creó
                if os.path.exists(test_path) and os.path.getsize(test_path) > 1000:
                    print(f"✅ {codec_name}: FUNCIONA")
                    working_codecs.append(codec_name)
                else:
                    print(f"❌ {codec_name}: Archivo inválido")
                
                # Limpiar archivo de prueba
                if os.path.exists(test_path):
                    os.remove(test_path)
            else:
                print(f"❌ {codec_name}: No se puede inicializar")
                
        except Exception as e:
            print(f"❌ {codec_name}: Error - {str(e)}")
    
    print(f"\n📊 RESULTADO: {len(working_codecs)} codecs funcionan")
    return working_codecs

def create_simple_test_video():
    """Crea un video de prueba simple."""
    print("\n🎬 CREANDO VIDEO DE PRUEBA...")
    print("-" * 40)
    
    try:
        width, height = 640, 480
        fps = 15
        duration = 3  # 3 segundos
        
        # Usar el directorio temporal del sistema
        output_path = os.path.join(tempfile.gettempdir(), "video_prueba_simple.avi")
        
        # Probar con XVID primero
        try:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            codec_used = "XVID"
        except:
            # Fallback a MJPG
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            codec_used = "MJPG"
        
        if not out.isOpened():
            return None, "❌ No se puede crear VideoWriter"
        
        print(f"📝 Usando codec: {codec_used}")
        print(f"📁 Creando: {output_path}")
        
        total_frames = fps * duration
        
        for frame_num in range(total_frames):
            # Frame simple con colores cambiantes
            color_value = int(255 * (frame_num / total_frames))
            frame = np.ones((height, width, 3), dtype=np.uint8) * color_value
            
            # Texto dinámico
            cv2.putText(frame, f"Frame {frame_num + 1}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255 - color_value, 0, 0), 3)
            cv2.putText(frame, f"Prueba Video", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255 - color_value, 0), 2)
            cv2.putText(frame, f"Codec: {codec_used}", (50, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255 - color_value), 2)
            
            # Círculo animado
            center_x = int(width/2 + 100 * np.sin(frame_num * 0.5))
            center_y = int(height/2)
            cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
            
            out.write(frame)
        
        out.release()
        
        # Verificar archivo
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Video creado exitosamente")
            print(f"📊 Tamaño: {file_size/1024:.1f} KB")
            print(f"📁 Ubicación: {output_path}")
            
            if file_size > 10000:  # Al menos 10KB
                return output_path, "✅ Video válido creado"
            else:
                return output_path, "⚠️ Video muy pequeño, posible problema"
        else:
            return None, "❌ Archivo no se creó"
            
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def test_video_reading(video_path):
    """Prueba si se puede leer el video creado."""
    print(f"\n📖 PROBANDO LECTURA DE VIDEO...")
    print("-" * 40)
    
    if not video_path or not os.path.exists(video_path):
        print("❌ Video no existe")
        return False
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ No se puede abrir el video")
            return False
        
        # Obtener propiedades
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Propiedades del video:")
        print(f"   Resolución: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Frames: {frame_count}")
        
        # Leer algunos frames
        frames_read = 0
        while frames_read < min(10, frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1
        
        cap.release()
        
        print(f"✅ Se leyeron {frames_read} frames correctamente")
        
        if frames_read > 0:
            return True
        else:
            print("❌ No se pudieron leer frames")
            return False
            
    except Exception as e:
        print(f"❌ Error leyendo video: {e}")
        return False

def main():
    """Función principal de diagnóstico."""
    print("🎬 DIAGNÓSTICO DE VIDEO - UPChiapas")
    print("="*50)
    print("Este script diagnostica problemas de video en tu sistema")
    print()
    
    # Paso 1: Probar codecs
    working_codecs = test_codecs()
    
    if not working_codecs:
        print("\n❌ PROBLEMA CRÍTICO: Ningún codec funciona")
        print("💡 SOLUCIONES POSIBLES:")
        print("   1. Reinstalar OpenCV: pip uninstall opencv-python && pip install opencv-python")
        print("   2. Instalar codecs adicionales: pip install opencv-contrib-python")
        print("   3. Verificar instalación de ffmpeg en tu sistema")
        return
    
    print(f"\n✅ Codecs disponibles: {', '.join(working_codecs)}")
    
    # Paso 2: Crear video de prueba
    video_path, status = create_simple_test_video()
    print(status)
    
    if not video_path:
        print("\n❌ PROBLEMA: No se puede crear videos")
        print("💡 POSIBLES CAUSAS:")
        print("   - Permisos de escritura insuficientes")
        print("   - Espacio en disco insuficiente")
        print("   - OpenCV mal instalado")
        return
    
    # Paso 3: Probar lectura
    if test_video_reading(video_path):
        print("\n🎉 ¡TODO FUNCIONA CORRECTAMENTE!")
        print("✅ Tu sistema puede crear y leer videos")
        print("✅ La aplicación debería funcionar perfectamente")
        
        print(f"\n📋 CONFIGURACIÓN RECOMENDADA:")
        print(f"   - Codec preferido: {working_codecs[0]}")
        print(f"   - Formato: .avi")
        print(f"   - FPS: 15-25")
        
    else:
        print("\n⚠️ PROBLEMA PARCIAL:")
        print("Se puede crear video pero no leer correctamente")
        print("💡 Intenta reinstalar OpenCV")
    
    # Limpiar
    if video_path and os.path.exists(video_path):
        os.remove(video_path)
        print(f"\n🧹 Archivo de prueba eliminado")
    
    print(f"\n🎯 PARA USAR EN TU APLICACIÓN:")
    print(f"   python app.py")
    print(f"   Los videos ahora deberían funcionar correctamente")

if __name__ == "__main__":
    main()