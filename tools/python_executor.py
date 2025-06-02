"""
Python Executor Tool
===================

This module implements safe Python code execution for the autonomous research agent,
allowing it to run generated code and capture outputs.
"""

import subprocess
import sys
import os
import tempfile
import time
from typing import Dict, Optional, List, Any
from pathlib import Path
import threading
import queue

class PythonExecutorTool:
    """
    Tool for executing Python code safely within the agent environment.
    """
    
    def __init__(self, project_root: str = ".", timeout: int = 300):
        """
        Initialize the Python executor tool.
        
        Args:
            project_root (str): Root directory for the project
            timeout (int): Maximum execution time in seconds
        """
        self.project_root = Path(project_root).resolve()
        self.timeout = timeout
        self.temp_dir = tempfile.gettempdir()
        
    def run_python_code(self, code: str, capture_output: bool = True,
                       working_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute Python code and return results.
        
        Args:
            code (str): Python code to execute
            capture_output (bool): Whether to capture stdout/stderr
            working_dir (str): Working directory for execution
            
        Returns:
            Dict: Execution results including output, errors, and return code
        """
        # Set working directory
        if working_dir:
            work_path = self.project_root / working_dir
        else:
            work_path = self.project_root
        
        # Create temporary file for the code
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False,
            dir=self.temp_dir
        )
        
        try:
            # Write code to temporary file
            temp_file.write(code)
            temp_file.flush()
            temp_file.close()
            
            # Execute the code
            result = self._execute_file(temp_file.name, work_path, capture_output)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Error preparing code execution: {str(e)}",
                "return_code": -1,
                "execution_time": 0
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def run_python_file(self, file_path: str, args: List[str] = None,
                       capture_output: bool = True) -> Dict[str, Any]:
        """
        Execute a Python file and return results.
        
        Args:
            file_path (str): Path to Python file
            args (List[str]): Command line arguments
            capture_output (bool): Whether to capture stdout/stderr
            
        Returns:
            Dict: Execution results
        """
        full_path = self.project_root / file_path
        
        if not full_path.exists():
            return {
                "success": False,
                "stdout": "",
                "stderr": f"File not found: {file_path}",
                "return_code": -1,
                "execution_time": 0
            }
        
        return self._execute_file(str(full_path), self.project_root, 
                                capture_output, args)
    
    def _execute_file(self, file_path: str, working_dir: Path,
                     capture_output: bool = True, 
                     args: List[str] = None) -> Dict[str, Any]:
        """
        Execute a Python file with subprocess.
        
        Args:
            file_path (str): Path to Python file
            working_dir (Path): Working directory
            capture_output (bool): Whether to capture output
            args (List[str]): Command line arguments
            
        Returns:
            Dict: Execution results
        """
        start_time = time.time()
        
        # Build command
        cmd = [sys.executable, file_path]
        if args:
            cmd.extend(args)
        
        try:
            # Set up environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            # Execute with timeout
            if capture_output:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(working_dir),
                    env=env
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout)
                    return_code = process.returncode
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                    return_code = -1
                    stderr += f"\nExecution timed out after {self.timeout} seconds"
            else:
                # Run without capturing output (for interactive scripts)
                process = subprocess.run(
                    cmd,
                    cwd=str(working_dir),
                    env=env,
                    timeout=self.timeout
                )
                stdout = ""
                stderr = ""
                return_code = process.returncode
            
            execution_time = time.time() - start_time
            
            return {
                "success": return_code == 0,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code,
                "execution_time": execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout} seconds",
                "return_code": -1,
                "execution_time": self.timeout
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "return_code": -1,
                "execution_time": execution_time
            }
    
    def run_python_script_with_input(self, code: str, input_data: str = "",
                                   capture_output: bool = True) -> Dict[str, Any]:
        """
        Execute Python code with stdin input.
        
        Args:
            code (str): Python code to execute
            input_data (str): Data to send to stdin
            capture_output (bool): Whether to capture output
            
        Returns:
            Dict: Execution results
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False,
            dir=self.temp_dir
        )
        
        try:
            temp_file.write(code)
            temp_file.flush()
            temp_file.close()
            
            start_time = time.time()
            
            # Execute with input
            process = subprocess.Popen(
                [sys.executable, temp_file.name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                text=True,
                cwd=str(self.project_root)
            )
            
            try:
                stdout, stderr = process.communicate(
                    input=input_data, 
                    timeout=self.timeout
                )
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -1
                if stderr:
                    stderr += f"\nExecution timed out after {self.timeout} seconds"
                else:
                    stderr = f"Execution timed out after {self.timeout} seconds"
            
            execution_time = time.time() - start_time
            
            return {
                "success": return_code == 0,
                "stdout": stdout or "",
                "stderr": stderr or "",
                "return_code": return_code,
                "execution_time": execution_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Error executing code with input: {str(e)}",
                "return_code": -1,
                "execution_time": 0
            }
        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def install_package(self, package_name: str) -> Dict[str, Any]:
        """
        Install a Python package using pip.
        
        Args:
            package_name (str): Name of package to install
            
        Returns:
            Dict: Installation results
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes for package installation
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "package": package_name
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Package installation timed out: {package_name}",
                "return_code": -1,
                "package": package_name
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Error installing package {package_name}: {str(e)}",
                "return_code": -1,
                "package": package_name
            }
    
    def check_syntax(self, code: str) -> Dict[str, Any]:
        """
        Check Python code syntax without executing it.
        
        Args:
            code (str): Python code to check
            
        Returns:
            Dict: Syntax check results
        """
        try:
            compile(code, '<string>', 'exec')
            return {
                "valid": True,
                "error": None
            }
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax error at line {e.lineno}: {e.msg}"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Compilation error: {str(e)}"
            }
    
    def format_execution_result(self, result: Dict[str, Any]) -> str:
        """
        Format execution results for the agent to process.
        
        Args:
            result (Dict): Execution result from run_python_code
            
        Returns:
            str: Formatted result string
        """
        formatted = "Python Execution Result:\n\n"
        
        formatted += f"**Success:** {result['success']}\n"
        formatted += f"**Return Code:** {result['return_code']}\n"
        formatted += f"**Execution Time:** {result['execution_time']:.2f} seconds\n\n"
        
        if result['stdout']:
            formatted += "**Output:**\n```\n"
            formatted += result['stdout']
            formatted += "\n```\n\n"
        
        if result['stderr']:
            formatted += "**Errors/Warnings:**\n```\n"
            formatted += result['stderr']
            formatted += "\n```\n\n"
        
        if not result['success']:
            formatted += "**Status:** Execution failed. Check errors above.\n"
        else:
            formatted += "**Status:** Execution completed successfully.\n"
        
        return formatted
    
    def run_interactive_session(self, commands: List[str]) -> Dict[str, Any]:
        """
        Run multiple Python commands in an interactive session.
        
        Args:
            commands (List[str]): List of Python commands
            
        Returns:
            Dict: Session results
        """
        # Create a script that runs all commands
        script_lines = []
        for i, cmd in enumerate(commands):
            script_lines.append(f"# Command {i+1}")
            script_lines.append(cmd)
            script_lines.append("")  # Empty line for separation
        
        script = "\n".join(script_lines)
        
        return self.run_python_code(script)

# Example usage and testing
if __name__ == "__main__":
    # Test the Python executor tool
    executor = PythonExecutorTool()
    
    # Test simple code execution
    test_code = """
import math
print("Hello from the research agent!")
print(f"Square root of 16: {math.sqrt(16)}")
result = 2 + 2
print(f"2 + 2 = {result}")
"""
    
    result = executor.run_python_code(test_code)
    print("Test Execution Result:")
    print(executor.format_execution_result(result))
    
    # Test syntax checking
    syntax_result = executor.check_syntax("print('valid syntax')")
    print(f"Syntax check (valid): {syntax_result}")
    
    syntax_result = executor.check_syntax("print('invalid syntax'")  # Missing closing paren
    print(f"Syntax check (invalid): {syntax_result}") 