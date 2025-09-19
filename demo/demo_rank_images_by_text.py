from clip_client import Client
from docarray import Document


def rank_images_by_text(text: str, image_uris: list) -> list:
    """
    使用CLIP模型根据文本描述对图像进行排序

    参数:
        text: 描述图像内容的文本
        image_uris: 图像文件的URI列表

    返回:
        排序后的结果列表，包含图像URI和对应的匹配分数
    """
    # 连接到CLIP服务器
    client = Client(server="grpc://0.0.0.0:61000")

    # 创建文档，包含文本和待匹配的图像
    doc = Document(text=text, matches=[Document(uri=uri) for uri in image_uris])

    # 执行排序操作
    result_doc = client.rank([doc])[0]  # 获取第一个文档结果

    # 提取并返回排序结果
    results = []
    for match in result_doc.matches:
        results.append({"uri": match.uri, "score": match.scores["clip_score"].value})

    # 按分数降序排序
    return sorted(results, key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    # 文本描述
    description = "many people in a conference room with a podium"

    # 待匹配的图像列表
    image_paths = [
        "test_pictures/绯村剑心.jpg",
        "test_pictures/演播室.png",
        "test_pictures/apple.jpeg",
    ]

    # 执行排序
    ranking_results = rank_images_by_text(description, image_paths)

    # 打印结果
    print("排序结果 (按匹配度从高到低):")
    for i, result in enumerate(ranking_results, 1):
        print(f"{i}. {result['uri']} - 匹配分数: {result['score']:.4f}")
