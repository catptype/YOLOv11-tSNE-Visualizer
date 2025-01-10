import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from sklearn.manifold import TSNE
from ultralytics import YOLO
from matplotlib import colormaps

class Yolo11Visualizer:
    """
    A visualizer for t-SNE analysis of features extracted from a YOLOv11 model.
    
    This class supports extracting features from the YOLOv11 model at the softmax layer 
    or raw logits using hooks, and visualizing the features using t-SNE.

    Attributes:
        __model (YOLO): The YOLOv11 model used for feature extraction.
        __seed (int): Random seed for reproducibility in t-SNE.
        __target_module (str): The specific model module name to attach the hook for extracting raw logits.
        __raw_logits (torch.Tensor): Stores raw logits extracted using a forward hook.
    """
    def __init__(self, model: str, seed: int = 42) -> None:
        """
        Initializes the Yolo11Visualizer.

        Args:
            model (str): Path to the YOLO model file.
            seed (int): Random seed for t-SNE and reproducibility. Defaults to 42.
        """
        self.__model = YOLO(model, task='classify')
        self.__seed = seed
        self.__target_module = "model.model.10.linear"
        self.__raw_logits = None
    
    def _tsne_features(self, features: np.ndarray, perplexity: int) -> np.ndarray:
        """
        Reduces the dimensionality of features using t-SNE.

        Args:
            features (np.ndarray): The features to be reduced.
            perplexity (int): Perplexity parameter for t-SNE.

        Returns:
            np.ndarray: 2D reduced feature space.
        """
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=self.__seed, n_jobs=-1)
        return tsne.fit_transform(features)

    def _extract_class_name(self, image_path: str) -> str:
        """
        Extracts the class name from the image path based on the directory structure.

        Directory Tree Structure:
            The function assumes that the class name is the name of the directory 
            immediately containing the image file. For example:

            root/
            ├── class1/
            │   ├── image1.jpg
            │   ├── image2.jpg
            ├── class2/
            │   ├── image3.jpg
            │   ├── image4.jpg

            If the `image_path` is "root/class1/image1.jpg", the extracted class name 
            will be "class1".

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Class name extracted from the directory name containing the image.
        """
        class_name = os.path.basename(os.path.dirname(image_path))
        return class_name

    def _plot_scatter(self, data: List[Tuple[str, np.ndarray]], perplexity: int, colormap_name: str) -> None:
        """
        Plots a scatter plot of t-SNE-reduced features grouped by class.

        Args:
            data (List[Tuple[str, np.ndarray]]): A list of tuples containing image paths and reduced features.
            perplexity (int): Perplexity parameter used in t-SNE.
            colormap_name (str): Name of the colormap to use for plotting.

        Raises:
            ValueError: If the specified colormap_name is not a valid colormap.

        Returns:
            None: Displays the scatter plot.
        """
        # Validate colormap
        if colormap_name not in plt.colormaps():
            raise ValueError(f"Invalid colormap '{colormap_name}'. Available colormaps: {plt.colormaps()}")

        class_points = {}
        for image_path, xy in data:
            class_name = self._extract_class_name(image_path)
            class_points.setdefault(class_name, []).append(xy)

        # Create a colormap and generate unique colors for each province
        num_classes = len(list(class_points.keys()))
        colormap = colormaps[colormap_name] # distinct colors

        # Plot each class with unique colors
        plot_output = plt.figure(figsize=(10, 7))
        for idx, (class_name, points) in enumerate(class_points.items()):
            points = np.array(points)
            plt.scatter(points[:, 0], points[:, 1], label=class_name, color=colormap(idx / num_classes), s=50, alpha=0.7)

        plt.legend(title="Classes", loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.title(f"t-SNE Scatter (perplexity={perplexity})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Save the plot as a JPEG file
        plot_output.savefig('scatter_plot.jpg', format='jpeg', dpi=150)
        print(f"Scatter plot saved as: scatter_plot.jpg")

    def _extract_features(self, image: np.ndarray, logit: bool) -> torch.Tensor:
        """
        Extracts features from the model at the raw logits or softmax layer.

        Args:
            image (np.ndarray): Input image in numpy array format.
            logit (bool): If True, extract raw logits using a hook. If False, extract softmax features.

        Returns:
            torch.Tensor: Extracted features from the model.
        """
        if logit:
            def hook(module, input, output):
                self.__raw_logits = output.detach()
           
            # Hook into the layer you want features from
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

    def visualize_tsne(self, image_paths: List[str], perplexity: int = 30, logit: bool = True, color: str = 'rainbow') -> None:
        """
        Visualizes the t-SNE scatter plot of features extracted from the model.

        Args:
            image_paths (List[str]): List of paths to image files.
            perplexity (int): Perplexity parameter for t-SNE. Defaults to 30.
            logit (bool): If True, use raw logits for visualization. If False, use softmax features. Defaults to True.
            color (str): Name of the colormap to use for plotting. Defaults to 'rainbow'.

            Recommended colormaps:
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
        features = []
        for path in image_paths:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            features.append(self._extract_features(image, logit))

        features = torch.cat(features).cpu().numpy()
        reduced_features = self._tsne_features(features, perplexity)
        self._plot_scatter(list(zip(image_paths, reduced_features)), perplexity, color)