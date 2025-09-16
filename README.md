# 安装依赖包
```shell
pip install uv
uv sync
```

# 启动服务端

```shell
JINA_LOG_LEVEL=DEBUG python -m clip_server search_flow.yaml
```