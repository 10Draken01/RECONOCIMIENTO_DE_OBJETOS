
import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import json
import matplotlib.pyplot as plt
from collections import Counter
import albumentations as A
from PIL import Image, ImageEnhance, ImageFilter

class DatasetUtils:
    """
    Utilidades para manejo del dataset
    """
    
    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = Path(dataset_path)
        self.stats = {}
    
    def validate_dataset(self) -> Dict:
        """
        Valida la integridad del dataset
        
        Returns:
            Dict: Estadísticas y errores encontrados
        """
        print("🔍 Validando dataset...")
        
        errors = []
        stats = {
            'train': {'images': 0, 'labels': 0, 'orphaned_images': [], 'orphaned_labels': []},
            'val': {'images': 0, 'labels': 0, 'orphaned_images': [], 'orphaned_labels': []},
            'test': {'images': 0, 'labels': 0, 'orphaned_images': [], 'orphaned_labels': []}
        }
        
        for split in ['train', 'val', 'test']:
            img_dir = self.dataset_path / "images" / split
            lbl_dir = self.dataset_path / "labels" / split
            
            if not img_dir.exists() or not lbl_dir.exists():
                errors.append(f"Directorio faltante para {split}")
                continue
            
            # Obtener archivos
            image_files = set([f.stem for f in img_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            label_files = set([f.stem for f in lbl_dir.glob("*.txt")])
            
            stats[split]['images'] = len(image_files)
            stats[split]['labels'] = len(label_files)
            
            # Encontrar archivos huérfanos
            stats[split]['orphaned_images'] = list(image_files - label_files)
            stats[split]['orphaned_labels'] = list(label_files - image_files)
            
            if stats[split]['orphaned_images']:
                errors.append(f"{split}: {len(stats[split]['orphaned_images'])} imágenes sin etiquetas")
            
            if stats[split]['orphaned_labels']:
                errors.append(f"{split}: {len(stats[split]['orphaned_labels'])} etiquetas sin imágenes")
        
        # Validar anotaciones
        annotation_errors = self._validate_annotations()
        errors.extend(annotation_errors)
        
        # Mostrar resultados
        self._print_validation_results(stats, errors)
        
        return {'stats': stats, 'errors': errors}
    
    def _validate_annotations(self) -> List[str]:
        """Valida el formato de las anotaciones YOLO"""
        errors = []
        
        for split in ['train', 'val', 'test']:
            lbl_dir = self.dataset_path / "labels" / split
            if not lbl_dir.exists():
                continue
                
            for label_file in lbl_dir.glob("*.txt"):
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split()
                        if len(parts) != 5:
                            errors.append(f"{label_file.name}:{line_num} - Formato incorrecto (debe tener 5 valores)")
                            continue
                        
                        try:
                            cls_id = int(parts[0])
                            x, y, w, h = map(float, parts[1:5])
                            
                            # Validar rangos
                            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                                errors.append(f"{label_file.name}:{line_num} - Coordenadas fuera de rango [0,1]")
                            
                            if cls_id != 0:  # Solo clase 0 para señales de alto
                                errors.append(f"{label_file.name}:{line_num} - ID de clase incorrecto (debe ser 0)")
                                
                        except ValueError:
                            errors.append(f"{label_file.name}:{line_num} - Valores no numéricos")
                            
                except Exception as e:
                    errors.append(f"Error leyendo {label_file.name}: {e}")
        
        return errors
    
    def _print_validation_results(self, stats: Dict, errors: List[str]):
        """Imprime los resultados de validación"""
        print("\n" + "="*60)
        print("📊 RESULTADOS DE VALIDACIÓN")
        print("="*60)
        
        # Estadísticas por split
        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()}:")
            print(f"  📸 Imágenes: {stats[split]['images']}")
            print(f"  🏷️  Etiquetas: {stats[split]['labels']}")
            
            if stats[split]['orphaned_images']:
                print(f"  ⚠️  Imágenes sin etiquetas: {len(stats[split]['orphaned_images'])}")
            
            if stats[split]['orphaned_labels']:
                print(f"  ⚠️  Etiquetas sin imágenes: {len(stats[split]['orphaned_labels'])}")
        
        # Totales
        total_images = sum(stats[split]['images'] for split in ['train', 'val', 'test'])
        total_labels = sum(stats[split]['labels'] for split in ['train', 'val', 'test'])
        
        print(f"\n📊 TOTALES:")
        print(f"  📸 Total imágenes: {total_images}")
        print(f"  🏷️  Total etiquetas: {total_labels}")
        
        # Errores
        if errors:
            print(f"\n❌ ERRORES ENCONTRADOS ({len(errors)}):")
            for error in errors[:10]:  # Mostrar solo los primeros 10
                print(f"  • {error}")
            if len(errors) > 10:
                print(f"  ... y {len(errors) - 10} errores más")
        else:
            print(f"\n✅ ¡Dataset válido! No se encontraron errores.")
        
        print("="*60)
    
    def split_dataset(self, source_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.2):
        """
        Divide el dataset en train/val/test
        
        Args:
            source_dir: Directorio con imágenes y etiquetas
            train_ratio: Proporción para entrenamiento
            val_ratio: Proporción para validación
        """
        source_path = Path(source_dir)
        
        # Obtener pares imagen-etiqueta
        image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        
        pairs = []
        for img_file in image_files:
            label_file = source_path / f"{img_file.stem}.txt"
            if label_file.exists():
                pairs.append((img_file, label_file))
        
        if not pairs:
            print("❌ No se encontraron pares imagen-etiqueta válidos")
            return
        
        # Mezclar y dividir
        random.shuffle(pairs)
        
        total = len(pairs)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        train_pairs = pairs[:train_count]
        val_pairs = pairs[train_count:train_count + val_count]
        test_pairs = pairs[train_count + val_count:]
        
        # Copiar archivos
        for split_name, split_pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
            img_dst = self.dataset_path / "images" / split_name
            lbl_dst = self.dataset_path / "labels" / split_name
            
            img_dst.mkdir(parents=True, exist_ok=True)
            lbl_dst.mkdir(parents=True, exist_ok=True)
            
            for img_file, lbl_file in split_pairs:
                shutil.copy2(img_file, img_dst / img_file.name)
                shutil.copy2(lbl_file, lbl_dst / lbl_file.name)
        
        print(f"✅ Dataset dividido:")
        print(f"  🏋️  Train: {len(train_pairs)} imágenes ({train_ratio*100:.1f}%)")
        print(f"  🎯 Val: {len(val_pairs)} imágenes ({val_ratio*100:.1f}%)")
        print(f"  🧪 Test: {len(test_pairs)} imágenes ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    def augment_dataset(self, target_count: int = 200):
        """
        Aumenta el dataset usando transformaciones
        
        Args:
            target_count: Número objetivo de imágenes de entrenamiento
        """
        train_img_dir = self.dataset_path / "images" / "train"
        train_lbl_dir = self.dataset_path / "labels" / "train"
        
        if not train_img_dir.exists():
            print("❌ Directorio de entrenamiento no encontrado")
            return
        
        # Obtener imágenes existentes
        existing_images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
        current_count = len(existing_images)
        
        if current_count >= target_count:
            print(f"✅ Ya tienes {current_count} imágenes (objetivo: {target_count})")
            return
        
        needed = target_count - current_count
        print(f"📈 Generando {needed} imágenes adicionales...")
        
        # Definir transformaciones
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        generated = 0
        while generated < needed:
            # Seleccionar imagen aleatoria
            source_img = random.choice(existing_images)
            source_lbl = train_lbl_dir / f"{source_img.stem}.txt"
            
            if not source_lbl.exists():
                continue
            
            # Leer imagen y etiquetas
            image = cv2.imread(str(source_img))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            bboxes, class_labels = self._read_yolo_labels(source_lbl)
            
            if not bboxes:
                continue
            
            try:
                # Aplicar transformaciones
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                # Guardar imagen aumentada
                new_name = f"{source_img.stem}_aug_{generated}"
                new_img_path = train_img_dir / f"{new_name}.jpg"
                new_lbl_path = train_lbl_dir / f"{new_name}.txt"
                
                # Guardar imagen
                transformed_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(new_img_path), transformed_image)
                
                # Guardar etiquetas
                self._write_yolo_labels(new_lbl_path, transformed['bboxes'], transformed['class_labels'])
                
                generated += 1
                
                if generated % 10 == 0:
                    print(f"  ✅ Generados: {generated}/{needed}")
                    
            except Exception as e:
                print(f"  ⚠️  Error procesando {source_img.name}: {e}")
                continue
        
        print(f"🎉 Dataset aumentado exitosamente! Total: {current_count + generated} imágenes")
    
    def _read_yolo_labels(self, label_path: Path) -> Tuple[List, List]:
        """Lee etiquetas en formato YOLO"""
        bboxes = []
        class_labels = []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        bboxes.append([x, y, w, h])
                        class_labels.append(cls_id)
        except Exception:
            pass
        
        return bboxes, class_labels
    
    def _write_yolo_labels(self, label_path: Path, bboxes: List, class_labels: List):
        """Escribe etiquetas en formato YOLO"""
        with open(label_path, 'w') as f:
            for bbox, cls_id in zip(bboxes, class_labels):
                x, y, w, h = bbox
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    
    def analyze_dataset(self):
        """Analiza y genera estadísticas del dataset"""
        print("📊 Analizando dataset...")
        
        analysis = {
            'bbox_sizes': [],
            'image_sizes': [],
            'objects_per_image': [],
            'class_distribution': Counter()
        }
        
        for split in ['train', 'val', 'test']:
            img_dir = self.dataset_path / "images" / split
            lbl_dir = self.dataset_path / "labels" / split
            
            if not img_dir.exists() or not lbl_dir.exists():
                continue
            
            for img_file in img_dir.glob("*"):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                # Analizar imagen
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
                    analysis['image_sizes'].append((w, h))
                
                # Analizar etiquetas
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    bboxes, class_labels = self._read_yolo_labels(lbl_file)
                    analysis['objects_per_image'].append(len(bboxes))
                    
                    for bbox, cls_id in zip(bboxes, class_labels):
                        _, _, w_norm, h_norm = bbox
                        analysis['bbox_sizes'].append((w_norm * w, h_norm * h))
                        analysis['class_distribution'][cls_id] += 1
        
        # Generar gráficas
        self._plot_analysis(analysis)
        
        return analysis
    
    def _plot_analysis(self, analysis: Dict):
        """Genera gráficas de análisis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis del Dataset de Señales de Alto', fontsize=16)
        
        # Distribución de tamaños de imagen
        if analysis['image_sizes']:
            widths, heights = zip(*analysis['image_sizes'])
            axes[0, 0].scatter(widths, heights, alpha=0.6)
            axes[0, 0].set_title('Tamaños de Imágenes')
            axes[0, 0].set_xlabel('Ancho (px)')
            axes[0, 0].set_ylabel('Alto (px)')
        
        # Distribución de tamaños de bounding boxes
        if analysis['bbox_sizes']:
            bbox_widths, bbox_heights = zip(*analysis['bbox_sizes'])
            axes[0, 1].scatter(bbox_widths, bbox_heights, alpha=0.6, color='orange')
            axes[0, 1].set_title('Tamaños de Bounding Boxes')
            axes[0, 1].set_xlabel('Ancho (px)')
            axes[0, 1].set_ylabel('Alto (px)')
        
        # Objetos por imagen
        if analysis['objects_per_image']:
            axes[1, 0].hist(analysis['objects_per_image'], bins=10, alpha=0.7, color='green')
            axes[1, 0].set_title('Objetos por Imagen')
            axes[1, 0].set_xlabel('Número de Objetos')
            axes[1, 0].set_ylabel('Frecuencia')
        
        # Distribución de clases
        if analysis['class_distribution']:
            classes, counts = zip(*analysis['class_distribution'].items())
            axes[1, 1].bar(['Señal Alto'], [counts[0]], color='red', alpha=0.7)
            axes[1, 1].set_title('Distribución de Clases')
            axes[1, 1].set_ylabel('Número de Instancias')
        
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Gráficas guardadas en: {self.dataset_path / 'dataset_analysis.png'}")

def main():
    """Función principal para utilidades del dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Utilidades para dataset de señales de alto")
    parser.add_argument('--validate', action='store_true', help='Validar dataset')
    parser.add_argument('--split', help='Dividir dataset desde directorio fuente')
    parser.add_argument('--augment', type=int, help='Aumentar dataset a N imágenes')
    parser.add_argument('--analyze', action='store_true', help='Analizar dataset')
    parser.add_argument('--dataset', default='dataset', help='Ruta al dataset')
    
    args = parser.parse_args()
    
    utils = DatasetUtils(args.dataset)
    
    if args.validate:
        utils.validate_dataset()
    
    if args.split:
        utils.split_dataset(args.split)
    
    if args.augment:
        utils.augment_dataset(args.augment)
    
    if args.analyze:
        utils.analyze_dataset()
    
    if not any(vars(args).values()):
        print("🛠️  Utilidades del Dataset - Señales de Alto")
        print("Usa --help para ver las opciones disponibles")

if __name__ == "__main__":
    main()