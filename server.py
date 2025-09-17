import base64
import io
import json
import os
import pathlib

import faiss
import numpy as np
from clip_client import Client
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from jina import Document, DocumentArray
from loguru import logger
from PIL import Image

# 初始化FastAPI应用
app = FastAPI(title="本地CLIP + 图像检索服务")

# 设置FAISS存储目录
BASE_PATH = pathlib.Path(__file__).parent.absolute()
FAISS_PATH = BASE_PATH.joinpath("faiss_data")
if not FAISS_PATH.exists():
    FAISS_PATH.mkdir(parents=True, exist_ok=True)

# FAISS索引和元数据存储路径
INDEX_PATH = FAISS_PATH.joinpath("image_index.faiss")
METADATA_PATH = FAISS_PATH.joinpath("image_metadata.json")

# 初始化CLIP客户端 - 连接到本地部署的CLIP服务
try:
    clip_client = Client("grpc://localhost:61000")
    print("成功连接到CLIP服务")
except Exception as e:
    print(f"初始化CLIP客户端失败: {str(e)}")
    clip_client = None
    raise SystemExit("请确保CLIP服务已启动并运行在grpc://localhost:61000")

# 向量维度 - ViT-B/32模型生成的向量维度是512
VECTOR_DIM = 512


# 初始化FAISS索引和元数据存储
class VectorDB:
    def __init__(self):
        self.index = None
        self.metadata = []
        self.load_index()

    def load_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(METADATA_PATH, "r") as f:
                self.metadata = json.load(f)
            print(f"加载索引成功，包含 {len(self.metadata)} 个图像")
        else:
            self.index = faiss.IndexFlatL2(VECTOR_DIM)
            self.metadata = []
            print("创建新的索引")

    def save_index(self):
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)
        print(f"索引已保存，共包含 {len(self.metadata)} 个图像")

    def add_vector(self, vector, metadata):
        vector_np = np.array(vector, dtype=np.float32).reshape(1, -1)
        self.index.add(vector_np)
        self.metadata.append(metadata)
        self.save_index()
        return len(self.metadata) - 1

    def search_similar(self, query_vector, top_k=5):
        if self.index.ntotal == 0:
            return []
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_np, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append(
                    {
                        "image_id": int(idx),
                        "distance": float(distances[0][i]),
                        "metadata": self.metadata[idx],
                    }
                )
        return results


# 初始化向量数据库
vector_db = VectorDB()


# 辅助函数：检查CLIP服务是否可用
def is_clip_available():
    if clip_client is None:
        return False
    try:
        # 使用DocumentArray进行测试，这是CLIP客户端最兼容的格式
        da = DocumentArray([Document(text="test")])
        result = clip_client.encode(da)
        return len(result) > 0 and len(result[0].embedding) == VECTOR_DIM
    except:
        return False


# 1. 添加图片到向量库的接口
@app.post("/images/add", summary="添加图片到向量库")
async def add_image(file: UploadFile = File(...), tags: str = None):
    if not is_clip_available():
        raise HTTPException(
            status_code=503, detail="CLIP服务不可用，请检查服务是否启动"
        )

    try:
        contents = await file.read()
        base64_encoded = base64.b64encode(contents).decode("utf-8")

        # 使用DocumentArray包装图像数据，这是最安全的方式
        da = DocumentArray([Document(uri=f"data:image/jpeg;base64,{base64_encoded}")])
        clip_client.encode(da, show_progress=False)

        # 从Document中提取嵌入向量
        image_embedding = da[0].embedding

        metadata = {"filename": file.filename, "tags": tags.split(",") if tags else []}

        image_id = vector_db.add_vector(image_embedding, metadata)

        return {
            "status": "success",
            "message": "图片已成功添加到向量库",
            "image_id": image_id,
            "storage_path": str(FAISS_PATH),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加图片失败: {str(e)}")


# 2. 语义文本比对检索接口
@app.get("/images/search", summary="通过文本搜索相似图片")
async def search_images(text: str, top_k: int = 5):
    if not is_clip_available():
        raise HTTPException(
            status_code=503, detail="CLIP服务不可用，请检查服务是否启动"
        )

    try:
        if not text:
            raise HTTPException(status_code=400, detail="搜索文本不能为空")

        # 使用DocumentArray处理文本
        da = DocumentArray([Document(text=text)])
        clip_client.encode(da, show_progress=False)
        text_embedding = da[0].embedding

        results = vector_db.search_similar(text_embedding, top_k)

        return {"status": "success", "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索图片失败: {str(e)}")


# 3. 图片智能打标签接口
@app.post("/images/tag", summary="为图片智能打标签")
async def tag_image(file: UploadFile = File(...), candidate_tags: str = None):
    if not is_clip_available():
        raise HTTPException(
            status_code=503, detail="CLIP服务不可用，请检查服务是否启动"
        )

    try:
        contents = await file.read()
        base64_encoded = base64.b64encode(contents).decode("utf-8")

        # 处理图像
        image_da = DocumentArray(
            [Document(uri=f"data:image/jpeg;base64,{base64_encoded}")]
        )
        clip_client.encode(image_da, show_progress=False)
        image_embedding = image_da[0].embedding

        # 处理候选标签
        if not candidate_tags:
            candidate_tags = [
                "animal",
                "person",
                "building",
                "car",
                "nature",
                "food",
                "city",
                "mountain",
                "ocean",
                "forest",
                "dog",
                "cat",
                "bird",
                "flower",
                "sunset",
                "night",
                "day",
                "indoor",
                "outdoor",
                "abstract",
            ]
        else:
            candidate_tags = candidate_tags.split(",")

        # 为每个标签生成向量
        tag_da = DocumentArray([Document(text=tag) for tag in candidate_tags])
        clip_client.encode(tag_da, show_progress=False)
        tag_embeddings = [doc.embedding for doc in tag_da]

        # 计算相似度
        similarities = []
        for i, tag in enumerate(candidate_tags):
            similarity = np.dot(image_embedding, tag_embeddings[i]) / (
                np.linalg.norm(image_embedding) * np.linalg.norm(tag_embeddings[i])
            )
            similarities.append((tag, float(similarity)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_tags = similarities[:5]

        return {
            "status": "success",
            "filename": file.filename,
            "tags": [
                {"tag": tag, "confidence": confidence} for tag, confidence in top_tags
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片打标签失败: {str(e)}")


# 健康检查接口
@app.get("/health", summary="服务健康检查")
async def health_check():
    logger.info("执行健康检查...")
    logger.info(clip_client.profile(content="test"))
    try:
        clip_available = is_clip_available()

        if clip_available:
            # 获取实际向量维度
            da = DocumentArray([Document(text="test")])
            clip_client.encode(da, show_progress=False)
            vector_dim = len(da[0].embedding)

            return {
                "status": "healthy" if clip_available else "unhealthy",
                "clip_service_available": clip_available,
                "message": "服务运行正常" if clip_available else "CLIP服务不可用",
                "vector_dimension": vector_dim,
                "expected_vector_dimension": VECTOR_DIM,
                "vector_dim_match": vector_dim == VECTOR_DIM,
                "faiss_storage_path": str(FAISS_PATH),
                "index_count": len(vector_db.metadata),
            }
        else:
            return {
                "status": "unhealthy",
                "clip_service_available": False,
                "message": "无法连接到CLIP服务，请检查服务是否在localhost:61000运行",
                "faiss_storage_path": str(FAISS_PATH),
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"健康检查失败: {str(e)}",
            "faiss_storage_path": str(FAISS_PATH),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
