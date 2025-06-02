"""
File Operations Tool
===================

This module implements safe file operations for the autonomous research agent,
including reading, writing, and managing files within the project directory.
"""

import os
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
import shutil
import tempfile

class FileOperationsTool:
    """
    Tool for safe file operations within the project directory.
    """
    
    def __init__(self, project_root: str = "."):
        """
        Initialize the file operations tool.
        
        Args:
            project_root (str): Root directory for the project
        """
        self.project_root = Path(project_root).resolve()
        self.allowed_extensions = {
            '.txt', '.md', '.py', '.json', '.csv', '.yaml', '.yml',
            '.tex', '.log', '.png', '.jpg', '.jpeg', '.pdf', '.html'
        }
        
    def _is_safe_path(self, file_path: str) -> bool:
        """
        Check if the file path is within the project directory and safe.
        
        Args:
            file_path (str): File path to check
            
        Returns:
            bool: True if path is safe
        """
        try:
            full_path = (self.project_root / file_path).resolve()
            return str(full_path).startswith(str(self.project_root))
        except:
            return False
    
    def _is_allowed_extension(self, file_path: str) -> bool:
        """
        Check if the file extension is allowed.
        
        Args:
            file_path (str): File path to check
            
        Returns:
            bool: True if extension is allowed
        """
        return Path(file_path).suffix.lower() in self.allowed_extensions
    
    def read_file(self, file_path: str, encoding: str = 'utf-8') -> Optional[str]:
        """
        Read content from a file.
        
        Args:
            file_path (str): Path to the file
            encoding (str): File encoding
            
        Returns:
            str: File content or None if error
        """
        if not self._is_safe_path(file_path):
            print(f"Unsafe file path: {file_path}")
            return None
        
        try:
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                print(f"File not found: {file_path}")
                return None
            
            with open(full_path, 'r', encoding=encoding) as f:
                return f.read()
                
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def write_file(self, file_path: str, content: str, 
                   encoding: str = 'utf-8', overwrite: bool = True) -> bool:
        """
        Write content to a file.
        
        Args:
            file_path (str): Path to the file
            content (str): Content to write
            encoding (str): File encoding
            overwrite (bool): Whether to overwrite existing files
            
        Returns:
            bool: True if successful
        """
        if not self._is_safe_path(file_path):
            print(f"Unsafe file path: {file_path}")
            return False
        
        if not self._is_allowed_extension(file_path):
            print(f"File extension not allowed: {file_path}")
            return False
        
        try:
            full_path = self.project_root / file_path
            
            if full_path.exists() and not overwrite:
                print(f"File exists and overwrite=False: {file_path}")
                return False
            
            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            print(f"Error writing file {file_path}: {str(e)}")
            return False
    
    def append_file(self, file_path: str, content: str, 
                    encoding: str = 'utf-8') -> bool:
        """
        Append content to a file.
        
        Args:
            file_path (str): Path to the file
            content (str): Content to append
            encoding (str): File encoding
            
        Returns:
            bool: True if successful
        """
        if not self._is_safe_path(file_path):
            print(f"Unsafe file path: {file_path}")
            return False
        
        if not self._is_allowed_extension(file_path):
            print(f"File extension not allowed: {file_path}")
            return False
        
        try:
            full_path = self.project_root / file_path
            
            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'a', encoding=encoding) as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            print(f"Error appending to file {file_path}: {str(e)}")
            return False
    
    def list_directory(self, dir_path: str = ".") -> List[str]:
        """
        List contents of a directory.
        
        Args:
            dir_path (str): Directory path
            
        Returns:
            List[str]: List of file/directory names
        """
        if not self._is_safe_path(dir_path):
            print(f"Unsafe directory path: {dir_path}")
            return []
        
        try:
            full_path = self.project_root / dir_path
            
            if not full_path.exists() or not full_path.is_dir():
                print(f"Directory not found: {dir_path}")
                return []
            
            return [item.name for item in full_path.iterdir()]
            
        except Exception as e:
            print(f"Error listing directory {dir_path}: {str(e)}")
            return []
    
    def create_directory(self, dir_path: str) -> bool:
        """
        Create a directory.
        
        Args:
            dir_path (str): Directory path to create
            
        Returns:
            bool: True if successful
        """
        if not self._is_safe_path(dir_path):
            print(f"Unsafe directory path: {dir_path}")
            return False
        
        try:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            return True
            
        except Exception as e:
            print(f"Error creating directory {dir_path}: {str(e)}")
            return False
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path (str): File path to check
            
        Returns:
            bool: True if file exists
        """
        if not self._is_safe_path(file_path):
            return False
        
        full_path = self.project_root / file_path
        return full_path.exists() and full_path.is_file()
    
    def directory_exists(self, dir_path: str) -> bool:
        """
        Check if a directory exists.
        
        Args:
            dir_path (str): Directory path to check
            
        Returns:
            bool: True if directory exists
        """
        if not self._is_safe_path(dir_path):
            return False
        
        full_path = self.project_root / dir_path
        return full_path.exists() and full_path.is_dir()
    
    def get_file_size(self, file_path: str) -> Optional[int]:
        """
        Get the size of a file in bytes.
        
        Args:
            file_path (str): File path
            
        Returns:
            int: File size in bytes or None if error
        """
        if not self._is_safe_path(file_path):
            return None
        
        try:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                return full_path.stat().st_size
            return None
            
        except Exception as e:
            print(f"Error getting file size {file_path}: {str(e)}")
            return None
    
    def copy_file(self, source_path: str, dest_path: str) -> bool:
        """
        Copy a file from source to destination.
        
        Args:
            source_path (str): Source file path
            dest_path (str): Destination file path
            
        Returns:
            bool: True if successful
        """
        if not self._is_safe_path(source_path) or not self._is_safe_path(dest_path):
            print("Unsafe file paths for copy operation")
            return False
        
        if not self._is_allowed_extension(dest_path):
            print(f"Destination file extension not allowed: {dest_path}")
            return False
        
        try:
            source_full = self.project_root / source_path
            dest_full = self.project_root / dest_path
            
            if not source_full.exists():
                print(f"Source file not found: {source_path}")
                return False
            
            # Create destination directory if needed
            dest_full.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source_full, dest_full)
            return True
            
        except Exception as e:
            print(f"Error copying file: {str(e)}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file.
        
        Args:
            file_path (str): File path to delete
            
        Returns:
            bool: True if successful
        """
        if not self._is_safe_path(file_path):
            print(f"Unsafe file path: {file_path}")
            return False
        
        try:
            full_path = self.project_root / file_path
            
            if full_path.exists() and full_path.is_file():
                full_path.unlink()
                return True
            else:
                print(f"File not found: {file_path}")
                return False
                
        except Exception as e:
            print(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    def read_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Read and parse a JSON file.
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            Dict: Parsed JSON data or None if error
        """
        content = self.read_file(file_path)
        if content is None:
            return None
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {file_path}: {str(e)}")
            return None
    
    def write_json(self, file_path: str, data: Dict[str, Any], 
                   indent: int = 2) -> bool:
        """
        Write data to a JSON file.
        
        Args:
            file_path (str): Path to JSON file
            data (Dict): Data to write
            indent (int): JSON indentation
            
        Returns:
            bool: True if successful
        """
        try:
            json_content = json.dumps(data, indent=indent, ensure_ascii=False)
            return self.write_file(file_path, json_content)
        except Exception as e:
            print(f"Error writing JSON file {file_path}: {str(e)}")
            return False
    
    def get_project_structure(self, max_depth: int = 3) -> Dict[str, Any]:
        """
        Get the project directory structure.
        
        Args:
            max_depth (int): Maximum depth to traverse
            
        Returns:
            Dict: Project structure
        """
        def _build_tree(path: Path, current_depth: int) -> Dict[str, Any]:
            if current_depth >= max_depth:
                return {"type": "directory", "truncated": True}
            
            if path.is_file():
                return {
                    "type": "file",
                    "size": path.stat().st_size,
                    "extension": path.suffix
                }
            elif path.is_dir():
                children = {}
                try:
                    for child in path.iterdir():
                        if not child.name.startswith('.'):  # Skip hidden files
                            children[child.name] = _build_tree(child, current_depth + 1)
                except PermissionError:
                    pass
                
                return {
                    "type": "directory",
                    "children": children
                }
            
            return {"type": "unknown"}
        
        return _build_tree(self.project_root, 0)

# Example usage and testing
if __name__ == "__main__":
    # Test the file operations tool
    file_ops = FileOperationsTool()
    
    # Test writing and reading
    test_content = "This is a test file for the research agent."
    success = file_ops.write_file("test_file.txt", test_content)
    print(f"Write test: {success}")
    
    if success:
        read_content = file_ops.read_file("test_file.txt")
        print(f"Read test: {read_content == test_content}")
        
        # Clean up
        file_ops.delete_file("test_file.txt") 