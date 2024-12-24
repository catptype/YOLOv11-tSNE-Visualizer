import os
import shutil
from typing import Tuple, List

class DirectoryProcessor:
    """
    Utility class for processing directories and files.

    Methods:
        get_all_files(directory, include_sub_dir=False): Retrieves a list of all files in a directory, including subdirectories.
        get_only_files(directory, extension, include_sub_dir=False): Retrieves a list of files with specified extensions in a directory, optionally including subdirectories.
        move_file(source, destination): Moves a file from the source path to the destination path.

    Example:
        ```python
        # Example usage of DirectoryProcessor class
        files = DirectoryProcessor.get_all_files(directory="/path/to/directory", include_sub_dir=True)
        image_files = DirectoryProcessor.get_only_files(directory="/path/to/images", extension=[".jpg", ".png"], include_sub_dir=False)
        DirectoryProcessor.move_file(source="/path/to/source/file.txt", destination="/path/to/destination/file.txt")
        ```

    """
    @staticmethod
    def get_dir(target_dir:str) -> List[str]:
        entries = os.listdir(target_dir)
        dir_list = [os.path.join(target_dir, entry) for entry in entries if os.path.isdir(os.path.join(target_dir, entry))]
        return dir_list

    @staticmethod
    def get_all_files(target_dir:str, include_sub_dir:bool=False) -> List[str]:
        if include_sub_dir:
            path_list = [os.path.join(root,file) for root, _, files in os.walk(target_dir, topdown=True) for file in files]
        else:
            path_list = [os.path.join(target_dir, file) for file in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, file))]

        return path_list

    @staticmethod
    def get_only_files(target_dir:str, extension:list, include_sub_dir:bool=False) -> List[str]:
        if include_sub_dir:
            path_list = [os.path.join(root,file) for root, _, files in os.walk(target_dir, topdown=True) for file in files]
            path_list = [path for path in path_list if any(ext in path for ext in extension)]
        else:
            path_list = [os.path.join(target_dir, file) for file in os.listdir(target_dir) if any(file.endswith(ext) for ext in extension)]

        return path_list
    
    @staticmethod
    def decompose_path(path:str) -> Tuple[str, str, str]:
        directory, filename = os.path.split(path)
        filename, extension = os.path.splitext(filename)
        return (directory, filename, extension)

    @staticmethod
    def path_up(path:str) -> Tuple[str, str]:
        parent_path = os.path.dirname(path)
        current_path = os.path.basename(path)
        return parent_path, current_path
    
    @staticmethod
    def create_dir(directory_path:str) -> None:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
    
    @staticmethod
    def move_file(source:str, destination:str) -> None:
        destination_dir = os.path.dirname(destination)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir, exist_ok=True)
        
        shutil.move(source, destination)

    @staticmethod
    def copy_file(source:str, destination:str) -> None:
        destination_dir = os.path.dirname(destination)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir, exist_ok=True)
        
        shutil.copyfile(source, destination)

    @staticmethod
    def rename_file(old_name:str, new_name:str) -> None:
        try:
            os.rename(old_name, new_name)
        except Exception as e:
            print(f"An error occurred: {e}")