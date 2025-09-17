from clip_client import Client
from docarray import Document, DocumentArray


def tag_image(image_path, candidate_labels):
    # 1. 连接到 CLIP 服务（确保服务已启动：clip-server run clip-flux --port 61000）
    c = Client("grpc://0.0.0.0:61000")

    # 2. 创建「主文档」：仅包含图像，不包含其他内容
    image_doc = Document(uri=image_path)  # 图像文档作为待匹配的主对象

    # 3. 创建「候选文档列表」：所有文本标签作为候选，存入主文档的 .matches 中
    candidate_docs = DocumentArray()
    for label in candidate_labels:
        text_doc = Document(text=label)  # 每个标签对应一个文本 Document
        candidate_docs.append(text_doc)

    # 关键步骤：将候选文本文档存入图像主文档的 .matches 属性
    image_doc.matches = candidate_docs

    # 4. 执行排序：仅传入主文档（图像），其 .matches 已包含所有候选标签
    # rank() 会计算主文档与自身 .matches 中所有候选的相似度
    ranked_docs = c.rank(
        docs=DocumentArray([image_doc]),  # 主文档列表（仅1个图像文档）
        top_k=len(candidate_labels),  # 返回所有候选的排序结果
        source="matches",  # 明确候选文档来自主文档的 .matches 属性
    )

    # 5. 提取排序结果（从主文档的 .matches 中获取分数）
    results = []
    for match in ranked_docs[0].matches:  # ranked_docs[0] 是排序后的图像主文档
        results.append(
            (match.text, match.scores["clip_score"].value)  # 候选文本标签  # 相似度分数
        )

    # 按相似度降序排列
    return sorted(results, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    # 图片路径（确保路径正确，支持绝对路径或相对路径）
    image_path = "test_pictures/演播室.png"  # 替换为你的图像路径

    # 候选标签列表（可根据需求扩展）
    candidate_labels = [
        "dog",
        "cat",
        "bird",
        "tree",
        "car",
        "building",
        "person",
        "mountain",
        "ocean",
        "flower",
        "computer",
        "phone",
        "apple",
        "banana",
        "fruit",
        "television",
        "studio",
    ]

    # 执行标签识别并打印结果
    try:
        results = tag_image(image_path, candidate_labels)
        print("图片标签及相似度（降序）:")
        for idx, (label, score) in enumerate(results, 1):
            print(f"{idx:2d}. {label:10s}: {score:.4f}")
    except Exception as e:
        print(f"处理失败: {str(e)}")
        print(
            "排查提示：1. CLIP服务是否启动？2. 图像路径是否正确？3. 依赖版本是否兼容？"
        )
