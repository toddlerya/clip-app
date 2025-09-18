import io
import tempfile
import uuid

from clip_client import Client
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# 初始化FastAPI应用
app = FastAPI(title="CLIP与Qdrant集成服务")

# 初始化CLIP客户端
clip_client = Client("grpc://localhost:61000")

# 初始化Qdrant客户端
qdrant_client = QdrantClient("localhost", port=6333)

# 定义集合名称
COLLECTION_NAME = "image_vectors"

# 检查集合是否存在，不存在则创建
if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )


def image_to_bytes(image: Image.Image) -> bytes:
    """将PIL Image对象转换为字节流"""
    img_byte_arr = io.BytesIO()
    # 保存为JPEG格式
    image.save(img_byte_arr, format="JPEG")
    # 获取字节数据
    return img_byte_arr.getvalue()


@app.post("/images", summary="将图片数据写入向量库")
async def add_image(
    file: UploadFile = File(..., description="要上传的图片文件"),
    tags: str = Query(None, description="图片标签，多个标签用逗号分隔"),
):
    try:
        # 读取图片文件
        image_data = await file.read()

        # 使用CLIP编码图片 - 直接使用字节数据
        image_embedding = clip_client.encode([image_data], is_image=True)[0]

        # 处理标签
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]

        # 生成唯一ID
        image_id = str(uuid.uuid4())

        # 存储到Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=image_id,
                    vector=image_embedding.tolist(),
                    payload={
                        "filename": file.filename,
                        "tags": tag_list,
                        "type": "image",
                    },
                )
            ],
        )

        return JSONResponse(
            content={
                "success": True,
                "message": "图片已成功添加到向量库",
                "data": {
                    "image_id": image_id,
                    "filename": file.filename,
                    "tags": tag_list,
                },
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "message": f"添加图片失败: {str(e)}"},
            status_code=500,
        )


@app.get("/search", summary="语义文本比对检索接口")
async def search_images(
    query: str = Query(..., description="检索用的文本"),
    tag: str = Query(None, description="可选的标签过滤"),
    limit: int = Query(10, description="返回结果数量"),
):
    try:
        # 使用CLIP编码文本
        text_embedding = clip_client.encode([query], is_image=False)[0]

        # 构建过滤条件
        filter_condition = None
        if tag:
            filter_condition = Filter(
                must=[FieldCondition(key="tags", match=MatchValue(value=tag))]
            )

        # 在Qdrant中搜索
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=text_embedding.tolist(),
            limit=limit,
            query_filter=filter_condition,
        )

        # 处理结果
        results = []
        for hit in search_result:
            results.append(
                {
                    "image_id": hit.id,
                    "filename": hit.payload.get("filename", ""),
                    "tags": hit.payload.get("tags", []),
                    "score": hit.score,
                }
            )

        return JSONResponse(content={"success": True, "data": results})
    except Exception as e:
        return JSONResponse(
            content={"success": False, "message": f"检索失败: {str(e)}"},
            status_code=500,
        )


@app.post("/images/tags", summary="基于语义文本智能打标签接口")
async def auto_tag_image(
    file: UploadFile = File(..., description="要打标签的图片文件"),
    candidate_tags: str = Query(..., description="候选标签，多个标签用逗号分隔"),
    top_n: int = Query(5, description="返回最相关的标签数量"),
):
    try:
        # 读取图片文件（直接使用字节数据）
        image_data = await file.read()

        # 使用CLIP编码图片
        image_embedding = clip_client.encode([image_data], is_image=True)[0]

        # 处理候选标签
        tags = [tag.strip() for tag in candidate_tags.split(",")]
        if not tags:
            return JSONResponse(
                content={"success": False, "message": "候选标签不能为空"},
                status_code=400,
            )

        # 编码候选标签
        tag_embeddings = clip_client.encode(tags, is_image=False)

        # 计算图片与每个标签的相似度（使用余弦相似度）
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity([image_embedding], tag_embeddings)[0]

        # 排序并选择最相关的标签
        tag_similarity = list(zip(tags, similarities))
        tag_similarity.sort(key=lambda x: x[1], reverse=True)

        # 获取前N个标签
        top_tags = tag_similarity[:top_n]

        return JSONResponse(
            content={
                "success": True,
                "data": {
                    "filename": file.filename,
                    "tags": [
                        {"tag": tag, "score": float(score)} for tag, score in top_tags
                    ],
                },
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "message": f"自动打标签失败: {str(e)}"},
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
