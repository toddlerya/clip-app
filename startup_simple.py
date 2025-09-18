# startup_simple.py - 简单启动版本
# 直接运行这个文件，不使用reload功能

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uvicorn

from clip_qdrant_server_claude import app

if __name__ == "__main__":
    print("Starting CLIP-Qdrant Service...")
    print("Make sure your CLIP server is running on localhost:61000")
    print("Make sure your Qdrant server is running on localhost:6333")
    print("Service will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)  # 关闭reload避免警告
