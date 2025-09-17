# 需求

基于开源算法研究和测试，需要支持私有化部署，提供接口给应用侧调用

接口：
1. 将图片数据写入向量库的接口；
2. 语义文本比对检索接口；
3. 基于语义文本智能打标签接口。

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