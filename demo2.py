from clip_client import Client
from docarray import Document

c = Client(server='grpc://0.0.0.0:61000')
r = c.rank(
    [
        Document(
            uri='test_pictures/绯村剑心.jpg',
            matches=[
                Document(text=f'这是一张{p}的照片')
                for p in (
                    '刀剑',
                    '苹果',
                    '水果',
                    '西瓜',
                    '男人',
                    '女人',
                    '风景',
                    '城市',
                    '电脑',
                    '手机',
                    '建筑',
                    '教室',
                    '会议室',
                    '办公室',
                    '武士'
                )
            ],
        )
    ]
)

print(r['@m', ['text', 'scores__clip_score__value']])