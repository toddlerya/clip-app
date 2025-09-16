# 安装依赖包
```shell
sudo update
sudo apt install build-essential g++
sudo apt install python3-dev
pip install uv
uv sync
```

# 启动服务端

```shell
JINA_LOG_LEVEL=DEBUG python -m clip_server search_flow.yaml
```