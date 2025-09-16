# https://clip-as-service.jina.ai/user-guides/retriever/
import time

from clip_client import Client
from docarray import Document

client = Client('grpc://0.0.0.0:61000')

# 1. 先验证编码功能是否正常
test_docs = [Document(text='test encoding'), Document(uri='https://clip-as-service.jina.ai/_static/favicon.png')]
encoded_docs = client.encode(test_docs)  # 手动调用编码接口

# 检查是否生成向量
for i, doc in enumerate(encoded_docs):
    if doc.embedding is None:
        print(f"❌ 文档{i}编码失败（无向量）")
    else:
        print(f"✅ 文档{i}编码成功，向量维度：{len(doc.embedding)}")  # 正常应为512

# 2. 编码正常后再执行索引和搜索
if all(doc.embedding is not None for doc in encoded_docs):
    # 索引文档
    client.index(
        [
            Document(text='she smiled, with pain'),
            # Document(uri='test_pictures/apple.jpg'),
            Document(uri='https://clip-as-service.jina.ai/_static/favicon.png'),
        ]
    )# 等待索引写入磁盘（AnnLiteIndexer默认异步写入，需短暂等待）
    import time
    time.sleep(3)

    # 执行搜索
    result = client.search(['smile'], limit=2)
    print("\n搜索结果:")
    print(result['@m', ['text', 'scores__cosine']])
else:
    print("❌ 编码失败，无法执行索引")