from clip_client import Client

c = Client("grpc://0.0.0.0:61000")
c.profile()

# r = c.encode(['First do it', 'then do it right', 'then do it better'])
# print(r)


r = c.encode(
    [
        "test_pictures/apple.jpeg",  # local image
        "https://clip-as-service.jina.ai/_static/favicon.png",  # remote image
        "data:image/gif;base64,R0lGODlhEAAQAMQAAORHHOVSKudfOulrSOp3WOyDZu6QdvCchPGolfO0o/XBs/fNwfjZ0frl3/zy7////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAkAABAALAAAAAAQABAAAAVVICSOZGlCQAosJ6mu7fiyZeKqNKToQGDsM8hBADgUXoGAiqhSvp5QAnQKGIgUhwFUYLCVDFCrKUE1lBavAViFIDlTImbKC5Gm2hB0SlBCBMQiB0UjIQA7",
    ]
)  # in image URI

print(r.shape)  # [3, 512]
