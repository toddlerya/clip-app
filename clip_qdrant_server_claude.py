# requirements.txt
"""
fastapi==0.104.1
uvicorn==0.24.0
jina==3.20.1
clip-server==0.8.7
qdrant-client==1.6.4
pillow==10.1.0
numpy==1.24.3
python-multipart==0.0.6
"""

import base64
import hashlib
import os
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from jina import Client, Document, DocumentArray
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# 预定义的标签库（可根据需要扩展）
GENERAL_TAGS = [
    "person",
    "people",
    "man",
    "woman",
    "child",
    "baby",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "dog",
    "cat",
    "bird",
    "horse",
    "cow",
    "sheep",
    "tree",
    "flower",
    "grass",
    "mountain",
    "water",
    "sky",
    "building",
    "house",
    "road",
    "bridge",
    "food",
    "fruit",
    "vegetable",
    "drink",
    "computer",
    "phone",
    "book",
    "chair",
    "table",
    "indoor",
    "outdoor",
    "nature",
    "city",
    "beach",
    "forest",
]


# 工具函数类
class FileUtils:
    """文件处理工具类"""

    @staticmethod
    def calculate_md5(data: bytes) -> str:
        """
        计算数据的MD5哈希值

        Args:
            data: 字节数据

        Returns:
            MD5哈希字符串
        """
        try:
            return hashlib.md5(data).hexdigest()
        except Exception as e:
            logger.error(f"计算MD5值失败: {e}")
            raise RuntimeError(f"Failed to calculate MD5: {str(e)}")

    @staticmethod
    def get_file_size(data: bytes) -> int:
        """
        获取文件大小（字节）

        Args:
            data: 字节数据

        Returns:
            文件大小（字节）
        """
        try:
            return len(data)
        except Exception as e:
            logger.error(f"获取文件大小失败: {e}")
            raise RuntimeError(f"Failed to get file size: {str(e)}")

    @staticmethod
    def generate_image_id(image_data: bytes) -> str:
        """
        基于图片数据生成唯一ID

        Args:
            image_data: 图片字节数据

        Returns:
            图片ID字符串
        """
        try:
            image_hash = FileUtils.calculate_md5(image_data)
            return f"img_{image_hash}"
        except Exception as e:
            logger.error(f"生成图片ID失败: {e}")
            raise RuntimeError(f"Failed to generate image ID: {str(e)}")

    @staticmethod
    def prepare_file_metadata(
        image_data: bytes,
        filename: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        准备文件元数据，包括文件名、大小、MD5等

        Args:
            image_data: 图片字节数据
            filename: 文件名
            custom_metadata: 自定义元数据

        Returns:
            完整的元数据字典
        """
        try:
            metadata = custom_metadata.copy() if custom_metadata else {}

            # 添加文件名
            if filename:
                metadata["file_name"] = filename

            # 添加文件大小
            metadata["file_size"] = FileUtils.get_file_size(image_data)

            # 添加MD5值
            metadata["md5"] = FileUtils.calculate_md5(image_data)

            return metadata

        except Exception as e:
            logger.error(f"准备文件元数据失败: {e}")
            raise RuntimeError(f"Failed to prepare file metadata: {str(e)}")


# Pydantic模型定义
class ImageUploadResponse(BaseModel):
    success: bool
    message: str
    image_id: str
    vector_id: Optional[str] = None


class SearchRequest(BaseModel):
    query_text: str
    top_k: int = 10
    filter_tags: Optional[List[str]] = None


class TagList(BaseModel):
    tags: List[str] = []


class ImageMetadata(BaseModel):
    filename: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    # 可以添加更多元数据字段


class SearchResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    total_count: int


class ImageListResponse(BaseModel):
    success: bool
    message: str
    images: List[Dict[str, Any]]
    total: int


class TaggingRequest(BaseModel):
    image_data: str  # base64编码的图片
    candidate_tags: Optional[List[str]] = None
    threshold: float = 0.5


class TaggingResponse(BaseModel):
    success: bool
    message: str
    tags: List[Dict[str, float]]  # {"tag": "cat", "score": 0.95}


class CLIPQdrantService:
    def __init__(
        self,
        clip_server_host: str = "localhost",
        clip_server_port: int = 61000,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "clip_images",
    ):
        """
        初始化CLIP-Qdrant服务

        Args:
            clip_server_host: CLIP服务器主机
            clip_server_port: CLIP服务器端口
            qdrant_host: Qdrant主机
            qdrant_port: Qdrant端口
            collection_name: 向量数据库集合名称
        """
        self.clip_client = Client(
            host=clip_server_host, port=clip_server_port, protocol="grpc"
        )
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name

        # 初始化集合
        self._init_collection()

        self.common_tags = GENERAL_TAGS

    def _init_collection(self):
        """初始化Qdrant集合"""
        try:
            # 检查集合是否存在
            collections = self.qdrant_client.get_collections()
            collection_exists = any(
                c.name == self.collection_name for c in collections.collections
            )

            if not collection_exists:
                # 创建集合，CLIP ViT-B/32 和 CN-CLIP/ViT-B-16的向量维度是512
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qdrant collection: {str(e)}")

    def encode_image(self, image_data: bytes) -> np.ndarray:
        """使用CLIP编码图片"""
        try:
            # 创建Document
            doc = Document(blob=image_data)
            doc_array = DocumentArray([doc])

            # 尝试不同的调用方式
            encoded_docs = None

            # 方法1: 直接调用
            try:
                encoded_docs = self.clip_client.post(on="/encode", inputs=doc_array)
            except Exception as e1:
                logger.info(f"Method 1 failed: {e1}")

                # 方法2: 使用空parameters
                try:
                    encoded_docs = self.clip_client.post(
                        on="/encode", inputs=doc_array, parameters={}
                    )
                except Exception as e2:
                    logger.info(f"Method 2 failed: {e2}")

                    # 方法3: 使用return_embeddings参数
                    try:
                        encoded_docs = self.clip_client.post(
                            on="/encode", inputs=doc_array, return_embeddings=True
                        )
                    except Exception as e3:
                        logger.info(f"Method 3 failed: {e3}")
                        raise RuntimeError(
                            f"All encoding methods failed. Last error: {e3}"
                        )

            # 提取向量
            if (
                encoded_docs
                and len(encoded_docs) > 0
                and encoded_docs[0].embedding is not None
            ):
                return encoded_docs[0].embedding
            else:
                raise ValueError("Failed to encode image - no embedding returned")

        except Exception as e:
            raise RuntimeError(f"Image encoding failed: {str(e)}")

    def encode_text(self, text: str) -> np.ndarray:
        """使用CLIP编码文本"""
        try:
            # 创建Document
            doc = Document(text=text)
            doc_array = DocumentArray([doc])

            # 尝试不同的调用方式
            encoded_docs = None

            # 方法1: 直接调用
            try:
                encoded_docs = self.clip_client.post(on="/encode", inputs=doc_array)
            except Exception as e1:
                logger.info(f"Method 1 failed: {e1}")

                # 方法2: 使用空parameters
                try:
                    encoded_docs = self.clip_client.post(
                        on="/encode", inputs=doc_array, parameters={}
                    )
                except Exception as e2:
                    logger.info(f"Method 2 failed: {e2}")

                    # 方法3: 使用return_embeddings参数
                    try:
                        encoded_docs = self.clip_client.post(
                            on="/encode", inputs=doc_array, return_embeddings=True
                        )
                    except Exception as e3:
                        logger.info(f"Method 3 failed: {e3}")
                        raise RuntimeError(
                            f"All encoding methods failed. Last error: {e3}"
                        )

            # 提取向量
            if (
                encoded_docs
                and len(encoded_docs) > 0
                and encoded_docs[0].embedding is not None
            ):
                return encoded_docs[0].embedding
            else:
                raise ValueError("Failed to encode text - no embedding returned")

        except Exception as e:
            raise RuntimeError(f"Text encoding failed: {str(e)}")

    def is_image_exist(self, image_id: str) -> bool:
        """检查图片是否已存在于向量库"""
        try:
            # 通过image_id过滤查询
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="image_id",  
                        match=MatchValue(value=image_id)
                    )
                ]
            )
            response, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1,
            )
            return len(response) > 0
        except Exception as e:
            logger.error(f"检查图片存在性失败: {e}")
            return False

    def add_image(
        self,
        image_data: bytes,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str]:
        """将图片添加到向量数据库"""
        message = "success"
        try:
            # 使用工具类生成图片ID
            image_id = FileUtils.generate_image_id(image_data)

            # 检查图片是否已存在
            if self.is_image_exist(image_id):
                message = f"图片已存在: {image_id}"
                logger.warning(message)
                # 直接返回已存在的image_id，不重复插入
                return message, image_id

            # 编码图片
            embedding = self.encode_image(image_data)

            # 准备payload
            payload = {
                "image_id": image_id,
                "tags": tags or [],
                "metadata": metadata or {},
            }

            # 创建Point并插入Qdrant
            point = PointStruct(
                id=str(uuid.uuid4()), vector=embedding.tolist(), payload=payload
            )

            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=[point]
            )

            return message, image_id

        except Exception as e:
            raise RuntimeError(f"Failed to add image: {str(e)}")

    def get_all_images(self) -> List[Dict[str, Any]]:
        """获取所有已入库的图片信息"""
        try:
            # 获取所有点
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # 设置一个合理的限制
            )

            # 处理结果
            results = []
            for point in scroll_result[0]:
                payload = point.payload
                results.append(
                    {
                        "image_id": payload.get("image_id", "unknown"),
                        "tags": payload.get("tags", []),
                        "metadata": payload.get("metadata", {}),
                        "vector_id": point.id,
                    }
                )

            return results
        except Exception as e:
            raise RuntimeError(f"Failed to get all images: {str(e)}")

    def search_by_text(
        self, query_text: str, top_k: int = 10, filter_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """根据文本搜索相似图片"""
        try:
            # 编码查询文本
            query_vector = self.encode_text(query_text)

            # 构建过滤条件
            query_filter = None
            if filter_tags:
                query_filter = Filter(
                    must=[
                        FieldCondition(key="tags", match=MatchValue(value=tag))
                        for tag in filter_tags
                    ]
                )

            # 搜索
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

            # 格式化结果
            results = []
            for result in search_results:
                results.append(
                    {
                        "vector_id": result.id,
                        "image_id": result.payload.get("image_id"),
                        "score": result.score,
                        "tags": result.payload.get("tags", []),
                        "metadata": result.payload.get("metadata", {}),
                    }
                )

            return results

        except Exception as e:
            raise RuntimeError(f"Search failed: {str(e)}")

    def generate_tags(
        self,
        image_data: bytes,
        candidate_tags: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> tuple[str, List[Dict[str, float]]]:
        """为图片生成智能标签"""
        message = "success"
        try:
            # 编码图片
            image_embedding = self.encode_image(image_data)

            # 使用候选标签或默认标签库
            tags_to_check = candidate_tags or self.common_tags

            # # 确保有标签可用
            # if not tags_to_check or len(tags_to_check) == 0:
            #     # 如果没有标签可用，使用一些基本标签
            #     tags_to_check = [
            #         "person",
            #         "animal",
            #         "cat",
            #         "dog",
            #         "bird",
            #         "building",
            #         "nature",
            #         "food",
            #         "vehicle",
            #         "indoor",
            #         "outdoor",
            #     ]
            #     logger.warning(f"没有找到标签库，使用默认标签: {tags_to_check}")

            # 计算图片与每个标签的相似度
            tag_scores = []

            for tag in tags_to_check:
                try:
                    # 编码标签文本
                    tag_embedding = self.encode_text(tag)

                    # 计算余弦相似度
                    similarity = np.dot(image_embedding, tag_embedding) / (
                        np.linalg.norm(image_embedding) * np.linalg.norm(tag_embedding)
                    )

                    # 降低阈值以确保能生成一些标签
                    # actual_threshold = min(threshold, 0.3)  # 确保阈值不高于0.3

                    # 只保留超过阈值的标签
                    if similarity > threshold:
                        tag_scores.append({"tag": tag, "score": float(similarity)})
                        logger.debug(f"标签 '{tag}' 相似度: {similarity:.4f} - 已添加")
                    else:
                        logger.debug(
                            f"标签 '{tag}' 相似度: {similarity:.4f} - 低于阈值"
                        )
                except Exception as tag_error:
                    logger.error(f"处理标签 '{tag}' 时出错: {str(tag_error)}")
                    continue

            # 按分数排序
            tag_scores.sort(key=lambda x: x["score"], reverse=True)

            # 如果没有标签超过阈值，返回前3个最相似的标签
            if not tag_scores and tags_to_check:
                message = "没有标签超过阈值, 返回前3个最相似的标签"
                logger.warning(message)
                all_scores = []
                for tag in tags_to_check:
                    try:
                        tag_embedding = self.encode_text(tag)
                        similarity = np.dot(image_embedding, tag_embedding) / (
                            np.linalg.norm(image_embedding)
                            * np.linalg.norm(tag_embedding)
                        )
                        all_scores.append({"tag": tag, "score": float(similarity)})
                    except Exception:
                        continue

                all_scores.sort(key=lambda x: x["score"], reverse=True)
                tag_scores = all_scores[:3]  # 返回前3个最相似的标签
                logger.info(f"返回的前3个标签: {tag_scores}")

            return message, tag_scores

        except Exception as e:
            message = f"标签生成失败: {str(e)}"
            logger.error(message)

            # 返回空列表而不是抛出异常，确保上传流程可以继续
            return message, []

    def get_existing_tags(self, image_id: str) -> List[str]:
        """获取已存在图片的标签"""
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="payload.image_id", match=MatchValue(value=image_id)
                    )
                ]
            )
            results = self.qdrant_client.scroll(
                collection_name=self.collection_name, filter=filter_condition, limit=1
            )
            if results[0]:
                return results[0][0].payload.get("tags", [])
            return []
        except Exception as e:
            logger.error(f"获取已有标签失败: {e}")
            return []


# 初始化服务
app = FastAPI(
    title="CLIP-Qdrant Image Service",
    description="基于CLIP和Qdrant的图像语义搜索服务",
    version="1.0.0",
)

# 初始化服务实例
try:
    service = CLIPQdrantService()
except Exception as e:
    logger.info(f"Failed to initialize service: {e}")
    service = None


@app.get("/")
async def root():
    """健康检查接口"""
    return {"message": "CLIP-Qdrant Service is running"}


@app.post("/api/v1/images/upload", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    tags: str = Form(None),  # 仍然接收表单数据
    metadata: str = Form(None),  # 仍然接收表单数据
):
    """
    上传图片到向量数据库

    Args:
        file: 上传的图片文件
        tags: 可选的标签列表（JSON字符串格式，如 '["cat", "animal"]'）
        metadata: 可选的元数据（JSON字符串格式，如 '{"filename": "cat.jpg", "author": "user1"}'）

    注意:
        系统会自动添加以下元数据字段:
        - file_name: 原始文件名
        - file_size: 文件大小(字节)
        - md5: 图片内容的MD5哈希值
    """
    if service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        # 验证文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # 读取图片数据
        image_data = await file.read()

        # 解析标签
        tag_list = None
        if tags:
            import json

            try:
                tag_list = json.loads(tags)
                # 确保tag_list是列表类型
                if not isinstance(tag_list, list):
                    tag_list = [str(tag_list)]
            except json.JSONDecodeError:
                # 如果不是JSON格式，当作逗号分隔的字符串处理
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        # 解析元数据
        metadata_dict = None
        if metadata:
            import json

            try:
                metadata_dict = json.loads(metadata)
                # 确保metadata_dict是字典类型
                if not isinstance(metadata_dict, dict):
                    metadata_dict = {"value": str(metadata_dict)}
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Metadata must be a valid JSON object"
                )

        # 使用工具类准备完整的文件元数据
        complete_metadata = FileUtils.prepare_file_metadata(
            image_data=image_data, filename=file.filename, custom_metadata=metadata_dict
        )

        # 添加图片
        message, image_id = service.add_image(image_data, tag_list, complete_metadata)

        return ImageUploadResponse(success=True, message=message, image_id=image_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/images", response_model=ImageListResponse)
async def list_images():
    """
    获取所有已入库的图片信息，包括向量ID、标签和元数据
    """
    if service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        images = service.get_all_images()

        return ImageListResponse(
            success=True,
            message="Images retrieved successfully",
            images=images,
            total=len(images),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/search/text", response_model=SearchResponse)
async def search_by_text(request: SearchRequest):
    """
    根据文本查询搜索相似图片

    Args:
        request: 搜索请求，包含查询文本、返回数量等
    """
    if service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        results = service.search_by_text(
            query_text=request.query_text,
            top_k=request.top_k,
            filter_tags=request.filter_tags,
        )

        return SearchResponse(success=True, results=results, total_count=len(results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/search/tags")
async def search_by_tags(tags: List[str], top_k: int = 10):
    """
    根据标签搜索图片

    Args:
        tags: 标签列表
        top_k: 返回结果数量
    """
    if service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        # 将标签组合成查询文本
        query_text = " ".join(tags)

        results = service.search_by_text(
            query_text=query_text, top_k=top_k, filter_tags=tags
        )

        return SearchResponse(success=True, results=results, total_count=len(results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/images/tagging", response_model=TaggingResponse)
async def auto_tagging(request: TaggingRequest):
    """
    图片智能打标签

    Args:
        request: 包含base64编码图片和可选候选标签的请求
    """
    if service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        # 解码base64图片
        try:
            image_data = base64.b64decode(request.image_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        # 验证是否为有效图片
        try:
            Image.open(BytesIO(image_data))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image format")

        image_id = FileUtils.generate_image_id(image_data)
        existing_tags = service.get_existing_tags(image_id)
        if existing_tags:
            return TaggingResponse(
                success=True, message="使用已有标签", tags=existing_tags
            )
        # 否则执行新的标签生成逻辑
        # 生成标签
        message, tags = service.generate_tags(
            image_data=image_data,
            candidate_tags=request.candidate_tags,
            threshold=request.threshold,
        )

        return TaggingResponse(success=True, message=message, tags=tags)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/images/upload_and_tag")
async def upload_and_tag(
    file: UploadFile = File(...),
    auto_tag: bool = Form(True),
    tag_threshold: float = Form(0.5),
    metadata: str = Form(None),  # 添加元数据参数
):
    """
    上传图片并自动打标签

    Args:
        file: 上传的图片文件
        auto_tag: 是否自动生成标签
        tag_threshold: 标签生成阈值
        metadata: 可选的元数据（JSON字符串格式，如 '{"filename": "cat.jpg", "author": "user1"}'）

    注意:
        系统会自动添加以下元数据字段:
        - file_name: 原始文件名
        - file_size: 文件大小(字节)
        - md5: 图片内容的MD5哈希值
    """
    if service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        # 验证文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # 读取图片数据
        image_data = await file.read()

        # 自动生成标签（如果启用）
        generated_tags = []
        tag_message = "success"
        tag_results = []

        if auto_tag:
            try:
                logger.info(f"开始生成标签，阈值: {tag_threshold}")
                tag_message, tag_results = service.generate_tags(
                    image_data, threshold=tag_threshold
                )
                logger.info(f"标签生成消息: {tag_message}")
                if tag_results:
                    generated_tags = [tag["tag"] for tag in tag_results]
                    logger.info(f"提取的标签: {generated_tags}")
                else:
                    logger.warning(f"没有生成任何标签: {tag_message}")
            except Exception as e:
                logger.error(f"标签生成过程中出错: {str(e)}")
                # 即使标签生成失败，也继续上传图片

        # 解析元数据
        metadata_dict = None
        if metadata:
            import json

            logger.info(f"上传图片，元数据: {metadata}")

            try:
                metadata_dict = json.loads(metadata)
                # 确保metadata_dict是字典类型
                if not isinstance(metadata_dict, dict):
                    metadata_dict = {"value": str(metadata_dict)}
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Metadata must be a valid JSON object"
                )

        # 使用工具类准备完整的文件元数据
        complete_metadata = FileUtils.prepare_file_metadata(
            image_data=image_data, filename=file.filename, custom_metadata=metadata_dict
        )

        # 添加图片到数据库
        message, image_id = service.add_image(
            image_data, generated_tags, complete_metadata
        )

        return {
            "success": True,
            "message": message,
            "image_id": image_id,
            "generated_tags": generated_tags,
            "tag_threshold": tag_threshold,
            "tag_message": tag_message,
            "tag_results": tag_results,
            "metadata": complete_metadata,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Starting CLIP-Qdrant Service...")
    logger.info("Make sure your CLIP server is running on localhost:61000")
    logger.info("Make sure your Qdrant server is running on localhost:6333")

    uvicorn.run("clip_qdrant_server_claude:app", host="0.0.0.0", port=8000, reload=True)
