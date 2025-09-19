# 需求

基于开源算法研究和测试，需要支持私有化部署，提供接口给应用侧调用

接口：
1. 将图片数据写入向量库的接口；
2. 语义文本比对检索接口；
3. 基于语义文本智能打标签接口。

# 部署
## 裸机部署
### 安装依赖包
```shell
sudo update
sudo apt install build-essential g++
sudo apt install python3-dev
pip install uv
uv sync
```

### 启动clip_server服务端

```shell
chmod +x start_server.sh
./start_server.sh
```

### 启动qdrant服务端
下载所需的安装包 https://github.com/qdrant/qdrant/releases 
比如 qdrant-x86_64-unknown-linux-musl.tar.gz
解压
```shell
tar -xzvf qdrant-x86_64-unknown-linux-musl.tar.gz
```
创建配置文件
```shell
mkdir -p qdrant/config
touch qdrant/config/production.yaml
```

配置文件内容
```
log_level: INFO
http_port: 6333
grpc_port: 6334
service_port: 6335
```

创建数据目录映射
```shell
mkdir -p qdrant/data
```

启动qdrant服务端
```shell
./qdrant --config-path qdrant/config/production.yaml                                
```

### 启动FastAPI服务

```shell
python clip_qdrant_server.py
``` 
