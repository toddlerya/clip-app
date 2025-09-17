import glob
import os
from typing import Dict, List, Tuple

import numpy as np
from clip_client import Client
from PIL import Image


class ClipImageSearcher:
    def __init__(self, server_addr: str = "grpc://0.0.0.0:61000"):
        """
        初始化CLIP图片检索器

        Args:
            server_addr: CLIP服务器地址
        """

        self.client = Client(server_addr)
        self.image_features = None  # 存储图片特征
        self.image_paths = []  # 存储图片路径

    def index_images(
        self, image_dir: str, extensions: Tuple[str] = (".jpg", ".jpeg", ".png", ".bmp")
    ) -> int:
        """
        索引指定目录下的所有图片

        Args:
            image_dir: 图片目录
            extensions: 支持的图片扩展名

        Returns:
            索引的图片数量
        """
        # 获取目录下所有符合条件的图片
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))

        if not self.image_paths:
            print(f"在目录 {image_dir} 中未找到图片")
            return 0

        print(f"开始索引 {len(self.image_paths)} 张图片...")

        # 提取图片特征 - 使用通用的encode方法并指定content_type
        self.image_features = self.client.encode(
            self.image_paths, content_type="image", batch_size=32
        )

        print(f"图片索引完成，共索引 {len(self.image_paths)} 张图片")
        return len(self.image_paths)

    def search(self, text: str, top_k: int = 5) -> List[Dict]:
        """
        根据文本描述搜索最匹配的图片

        Args:
            text: 搜索文本
            top_k: 返回最匹配的前k张图片

        Returns:
            包含匹配图片路径和相似度分数的字典列表
        """
        if self.image_features is None or len(self.image_paths) == 0:
            raise ValueError("请先调用index_images方法索引图片")

        # 编码文本 - 使用通用的encode方法并指定content_type
        text_feature = self.client.encode([text], content_type="text")[0]

        # 计算文本特征与所有图片特征的余弦相似度
        similarities = self._cosine_similarity(text_feature, self.image_features)

        # 获取相似度最高的前k张图片索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # 从高到低排序

        # 准备结果
        results = []
        for idx in top_indices:
            results.append(
                {
                    "image_path": self.image_paths[idx],
                    "similarity": float(similarities[idx]),
                }
            )

        return results

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b, axis=1)
        return np.dot(b, a) / (a_norm * b_norm)


# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="使用CLIP进行图片语义检索")
    parser.add_argument(
        "--server", type=str, default="grpc://0.0.0.0:61000", help="CLIP服务器地址"
    )
    parser.add_argument("--image-dir", type=str, required=True, help="图片目录")
    parser.add_argument("--text", type=str, required=True, help="搜索文本")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数量")

    args = parser.parse_args()

    # 初始化检索器
    searcher = ClipImageSearcher(args.server)

    # 索引图片
    searcher.index_images(args.image_dir)

    # 执行搜索
    results = searcher.search(args.text, args.top_k)

    # 打印结果
    print(f"\n与 '{args.text}' 最匹配的图片:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['image_path']} (相似度: {result['similarity']:.4f})")
