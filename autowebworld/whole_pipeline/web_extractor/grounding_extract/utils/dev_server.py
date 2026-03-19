"""Vue development server manager."""

import subprocess
import time
import socket
import re
import threading
from pathlib import Path
from typing import Optional, Tuple


class DevServer:
    """Manage Vue development server (npm run dev)."""

    def __init__(self, web_dir: str, port: int = 0):
        self.web_dir = Path(web_dir)
        if not self.web_dir.exists():
            raise FileNotFoundError(f"Web directory not found: {web_dir}")

        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.actual_port: Optional[int] = None
        self.base_url: Optional[str] = None
        self.output_thread: Optional[threading.Thread] = None
        self.server_ready = threading.Event()

    def start(self, timeout: int = 60) -> str:
        """Start dev server and return URL.
        
        Args:
            timeout: Maximum seconds to wait for server to start
            
        Returns:
            Base URL of the started server
            
        Raises:
            RuntimeError: If server fails to start within timeout
        """
        if self.process:
            return self.base_url

        # Find free port if not specified
        if self.port == 0:
            self.actual_port = self._find_free_port()
        else:
            self.actual_port = self.port

        # Check if package.json exists
        package_json = self.web_dir / "package.json"
        if not package_json.exists():
            raise FileNotFoundError(f"package.json not found in {self.web_dir}")

        # Start npm run dev with specified port
        env = {
            **subprocess.os.environ,
            "PORT": str(self.actual_port),
            "VITE_PORT": str(self.actual_port),
        }

        # Start process
        self.process = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(self.actual_port), "--host"],
            cwd=str(self.web_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        # Start thread to read output and detect when server is ready
        self.output_thread = threading.Thread(
            target=self._read_output,
            daemon=True
        )
        self.output_thread.start()

        # Wait for server to be ready
        if not self.server_ready.wait(timeout=timeout):
            self.stop()
            raise RuntimeError(f"Dev server failed to start within {timeout} seconds")

        self.base_url = f"http://localhost:{self.actual_port}"
        return self.base_url

    def stop(self):
        """Stop the dev server."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            finally:
                self.process = None
                self.actual_port = None
                self.base_url = None
                self.server_ready.clear()

    def _read_output(self):
        """Read process output and detect when server is ready."""
        if not self.process or not self.process.stdout:
            return

        # Patterns to detect server ready
        # Vite typically outputs: "Local:   http://localhost:5173/"
        ready_patterns = [
            r"Local:\s+http://localhost:(\d+)",
            r"http://localhost:(\d+)",
            r"ready in \d+ms",
            r"server running at",
        ]

        for line in self.process.stdout:
            # Check if server is ready
            for pattern in ready_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Extract port if available
                    if match.groups():
                        detected_port = int(match.group(1))
                        if self.actual_port is None:
                            self.actual_port = detected_port
                    
                    # Signal that server is ready
                    self.server_ready.set()
                    break

            # Continue reading to prevent buffer overflow
            # but don't print to avoid cluttering output

    def _find_free_port(self) -> int:
        """Find a free port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def is_running(self) -> bool:
        """Check if server is running."""
        return self.process is not None and self.process.poll() is None


def start_dev_server(web_dir: str, port: int = 0, timeout: int = 60) -> Tuple:
    """Start Vue dev server.
    
    Args:
        web_dir: Path to Vue project directory
        port: Port to use (0 for auto-assign)
        timeout: Maximum seconds to wait for server to start
        
    Returns:
        Tuple of (server, url)
    """
    server = DevServer(web_dir, port)
    url = server.start(timeout=timeout)
    return server, url

