from clip_client import Client
from docarray import Document, DocumentArray

def tag_image(image_path, candidate_labels):
    # 连接到 CLIP 服务
    c = Client("grpc://0.0.0.0:61000")

    # 创建图像文档并初始化matches
    image_doc = Document(uri=image_path)
    image_doc.matches = DocumentArray()  # 初始化matches

    # 创建文本文档列表并初始化matches
    text_docs = DocumentArray()
    for label in candidate_labels:
        doc = Document(text=label)
        doc.matches = DocumentArray()  # 初始化matches
        text_docs.append(doc)

    # 合并所有文档
    docs = DocumentArray([image_doc]) + text_docs

    # 计算相似度
    similarities = c.rank(docs, top_k=len(candidate_labels))

    # 提取结果并排序
    results = []
    for match in similarities[0].matches:  # 第一个文档是图像
        results.append((match.text, match.scores['clip_score'].value))

    return sorted(results, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    # 图片路径
    image_path = "test_pictures/apple.jpg"  # 替换为你的图片路径

    # 候选标签列表（可以根据你的需求扩展）
    candidate_labels = [
        "dog", "cat", "bird", "tree", "car", "building",
        "person", "mountain", "ocean", "flower", "computer", "phone", "apple", "banana", "fruit"
    ]

    # 获取标签结果
    try:
        # 获取标签结果
        results = tag_image(image_path, candidate_labels)

        # 打印结果
        print("图片标签及相似度:")
        for label, score in results:
            print(f"{label}: {score:.4f}")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")