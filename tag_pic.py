from clip_client import Client
from docarray import Document, DocumentArray

# --------------------------
# 1. 通用标签池（可扩展至数千个）
# --------------------------
GENERIC_LABELS = [
    # 物体类
    "comics",
    "anime",
    "cartoon",
    "people",
    "warrior",
    "wizard",
    "knight",
    "hero",
    "villain",
    "robot",
    "male",
    "female",
    "child",
    "adult",
    "tree",
    "flower",
    "grass",
    "sword",
    "shield",
    "apple",
    "banana",
    "orange",
    "grape",
    "pear",
    "watermelon",
    "strawberry",
    "cat",
    "dog",
    "bird",
    "fish",
    "rabbit",
    "mouse",
    "tiger",
    "lion",
    "car",
    "bus",
    "bike",
    "train",
    "plane",
    "ship",
    "truck",
    "computer",
    "phone",
    "tv",
    "laptop",
    "keyboard",
    "mouse",
    "speaker",
    "chair",
    "table",
    "sofa",
    "bed",
    "desk",
    "book",
    "pen",
    "bag",
    # 场景类
    "forest",
    "mountain",
    "ocean",
    "river",
    "lake",
    "desert",
    "city",
    "village",
    "room",
    "office",
    "school",
    "hospital",
    "restaurant",
    "shop",
    # 属性类
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "gray",
    "round",
    "square",
    "long",
    "short",
    "big",
    "small",
    "new",
    "old",
]


# --------------------------
# 2. 自动打标函数（无候选标签输入）
# --------------------------
def auto_tag_image(image_path, top_k=5):
    # 连接CLIP服务（确保服务已启动：clip-server run clip-vit-base-patch32 --port 61000）
    c = Client("grpc://0.0.0.0:61000")

    # 创建图像主文档（待打标的对象）
    image_doc = Document(uri=image_path)

    # 构建标签池对应的候选文档（自动从通用标签池生成）
    candidate_docs = DocumentArray([Document(text=label) for label in GENERIC_LABELS])
    image_doc.matches = candidate_docs  # 将候选标签存入主文档的matches

    # CLIP计算相似度并排序
    ranked_docs = c.rank(
        docs=DocumentArray([image_doc]),
        top_k=len(GENERIC_LABELS),  # 先计算所有标签的相似度
        source="matches",
    )

    # 提取Top-K高相似度标签
    results = [
        (match.text, match.scores["clip_score"].value)
        for match in ranked_docs[0].matches[:top_k]  # 取前N个结果
    ]

    return results


# --------------------------
# 3. 运行示例
# --------------------------
if __name__ == "__main__":
    image_path = "test_pictures/绯村剑心.jpg"  # 替换为你的图像路径
    try:
        # 自动打标，返回Top-5标签
        tags = auto_tag_image(image_path, top_k=5)
        print("自动生成的图像标签（相似度降序）:")
        for idx, (label, score) in enumerate(tags, 1):
            print(f"{idx}. {label:10s} | 相似度: {score:.4f}")
    except Exception as e:
        print(f"打标失败: {str(e)}")
        print("排查提示：1. CLIP服务是否启动？2. 图像路径是否正确？")
