from clip_client import Client
from docarray import Document

c = Client(server='grpc://0.0.0.0:61000')
r = c.rank(
    [
        Document(
            uri='test_pictures/rerank.png',
            matches=[
                Document(text=f'a photo of a {p}')
                for p in (
                    'control room',
                    'lecture room',
                    'conference room',
                    'podium indoor',
                    'television studio',
                )
            ],
        )
    ]
)

print(r['@m', ['text', 'scores__clip_score__value']])