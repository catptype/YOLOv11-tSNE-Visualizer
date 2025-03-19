import cv2
import os
import threading
import torch
import matplotlib.pyplot as plt
import numpy as np

from typing import List
from sklearn.manifold import TSNE
from ultralytics import YOLO
from matplotlib import colormaps
from queue import Queue
from itertools import combinations

from util.TextProgressBar import TextProgressBar

class Yolo11Visualizer:
    def __init__(self, model:str, seed:int=42) -> None:
        self.__model = YOLO(model, task='classify')
        _ = self.__model(np.zeros((1, 1, 3), dtype=np.uint8), verbose=False)
        self.__seed = seed
        self.__target_module = "model.model.10.linear"
        self.__raw_logits = None
        self.__cls_pts = None
    
    def _extract_features(self, image:np.ndarray, logit:bool) -> torch.Tensor:
        if logit:
            def hook(module, input, output):
                self.__raw_logits = output.detach()

            # Register layer to hook
            for name, module in self.__model.named_modules():
                if name == self.__target_module:
                    module.register_forward_hook(hook)
                    break
            
            # Perform a forward pass
            _ = self.__model(image, verbose=False)

            return self.__raw_logits
        else:
            # Directly use the model's prediction or embedding layer output
            with torch.no_grad():
                return self.__model(image, embed=[-1], verbose=False)[0]
    
    def calculate_tsne(self, image_paths: List[str], perplexity:int=30, logit:bool=True, worker:int=4):
        prog_bar = TextProgressBar(len(image_paths))

        feature_list = []
        thread_queue = Queue()
        worker = min(worker, os.cpu_count())
        
        def execute(path):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            feature_list.append(self._extract_features(image, logit))
            prog_bar.add_step()

        for path in image_paths:
            # Wait for an available worker if the queue is full
            while thread_queue.qsize() >= worker:
                oldest_thread = thread_queue.get()
                oldest_thread.join()
            
            # Start a new thread for each path
            thread = threading.Thread(target=execute, args=(path,), daemon=True)
            thread.start()
            thread_queue.put(thread)
        
        # Ensure all threads have completed
        while not thread_queue.empty():
            thread = thread_queue.get()
            thread.join()
        
        # Convert from tensor to numpy
        features = torch.cat(feature_list).cpu().numpy()

        # Perform reduce feature
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=self.__seed, n_jobs=-1)
        reduced_features = tsne.fit_transform(features)

        # Bind data
        self.__cls_pts = []
        for path, xy in list(zip(image_paths, reduced_features)):
            cls_name = os.path.basename(os.path.dirname(path))
            self.__cls_pts.append((path, cls_name, tuple(map(float, xy))))
        
        print("COMPLETE")
        print("call property 'cls_pts' to retrieve class points")
    
    def plot_tsne(self, colormap_name: str, title: str="t-SNE Scatter", export: bool = False):
        """
        ```markdown
        - 'rainbow': A colorful gradient that works well for distinguishing classes.
        - 'viridis': A perceptually uniform colormap suitable for various types of data.
        - 'plasma': A smooth gradient with warm colors.
        - 'cool': A simple blue-to-pink gradient.
        - 'spring': A gradient with pink and yellow hues.
        - 'summer': A green-to-yellow gradient.
        - 'autumn': A gradient with warm orange and yellow tones.
        ```
        """
        # Validate colormap
        if colormap_name not in plt.colormaps():
            raise ValueError(f"Invalid colormap '{colormap_name}'. Available colormaps: {plt.colormaps()}")
        
        # Convert full data into dictionary with class names as keys
        class_points = {}
        for _, class_name, coordinates in self.__cls_pts:
            class_points.setdefault(class_name, []).append(coordinates)
        
        # Determine number of unique classes
        num_classes = len(class_points)
        cmap = colormaps[colormap_name]  # Retrieve colormap
        
        # Plot each class with unique colors
        fig = plt.figure(figsize=(10, 7))
        for idx, (class_name, points) in enumerate(class_points.items()):
            points = np.array(points)
            plt.scatter(points[:, 0], points[:, 1], label=class_name, color=cmap(idx / num_classes), s=50, alpha=0.7)
        
        plt.legend(title="Classes", loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        if export:
            os.makedirs("result", exist_ok=True)
            image_path = os.path.join("result", f"t-SNE Scatter.jpg")
            text_path = os.path.join("result", f"t-SNE Scatter.txt")
            fig.savefig(image_path, format='jpeg', dpi=150)
            
            with open(text_path, "w") as file:
                sorted_data = sorted(self.__cls_pts, key=lambda x: (x[2][0], x[2][1]))
                for path, label, coords in sorted_data:
                    file.write(f"{coords} | {label} | {path}\n")

    def plot_compare_tsne(self, colormap_name:str="rainbow", num_classes:int=2, export:bool=False):
        """
        ```markdown
        - 'rainbow': A colorful gradient that works well for distinguishing classes.
        - 'viridis': A perceptually uniform colormap suitable for various types of data.
        - 'plasma': A smooth gradient with warm colors.
        - 'cool': A simple blue-to-pink gradient.
        - 'spring': A gradient with pink and yellow hues.
        - 'summer': A green-to-yellow gradient.
        - 'autumn': A gradient with warm orange and yellow tones.
        ```
        """
        # Validate num_classes
        if num_classes < 2:
            raise ValueError("num_classes should be >= 2")
        
        # Validate colormap
        if colormap_name not in plt.colormaps():
            raise ValueError(f"Invalid colormap '{colormap_name}'. Available colormaps: {plt.colormaps()}")
        
        # Convert data into dictionary with class names as keys
        class_points = {}
        for _, class_name, coordinates in self.__cls_pts:
            class_points.setdefault(class_name, []).append(coordinates)
        
        # Generate class combinations
        class_combinations = list(combinations(class_points.keys(), num_classes))
        cmap = colormaps[colormap_name]  # Retrieve colormap
        
        # Plot comparison of selected class pairs
        for class_pair in class_combinations:
            fig = plt.figure(figsize=(10, 7))
            title = " vs ".join(list(class_pair))
            export_data = []
            
            for idx, class_name in enumerate(class_pair):
                points = np.array(class_points[class_name])
                export_data.extend([entry for entry in self.__cls_pts if entry[1] == class_name])
                plt.scatter(points[:, 0], points[:, 1], label=class_name, color=cmap(idx / num_classes), s=50, alpha=0.7)
            
            plt.legend(title="Classes", loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.title(title.strip())
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            if export:
                os.makedirs("compare", exist_ok=True)
                image_path = os.path.join("compare", f"{title.strip()}.jpg")
                text_path = os.path.join("compare", f"{title.strip()}.txt")
                fig.savefig(image_path, format='jpeg', dpi=150)
                
                with open(text_path, "w") as file:
                    sorted_data = sorted(export_data, key=lambda x: (x[2][0], x[2][1]))
                    for path, label, coords in sorted_data:
                        file.write(f"{coords} | {label} | {path}\n")

    @property
    def cls_pts(self):
        return self.__cls_pts