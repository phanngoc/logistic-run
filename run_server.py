#!/usr/bin/env python3
"""Script khởi chạy MVP server"""

import sys
import os
import subprocess
import time

def start_server():
    """Khởi chạy server"""
    print("\nStarting MVP Dispatch Optimization Server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000", 
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

def main():
    """Main function"""
    print("=== MVP Dispatch Optimization Setup ===")
    
    # Check if we're in the right directory
    if not os.path.exists("app/main.py"):
        print("Error: Please run this script from the project root directory")
        sys.exit(1)

    # Start server
    start_server()

if __name__ == "__main__":
    main()
