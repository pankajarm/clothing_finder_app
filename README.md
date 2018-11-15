# Cloth Finder App
From Deep Learning to Web App in 50 Lines of Code

```
from starlette.applications import Starlette
from starlette.responses import RedirectResponse, FileResponse
from starlette.routing import Router, Mount
from starlette.staticfiles import StaticFiles
from fastai.vision import ImageDataBunch, create_cnn, open_image, get_transforms, models, imagenet_stats
import torch, sys, uvicorn, aiohttp
from pathlib import Path
from io import BytesIO

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

app = Router(routes=[Mount('/static', app=StaticFiles(directory='static')),])
path = Path('data/cloth_categories')
classes = ['Blouse', 'Blazer', 'Button-Down', 'Bomber', 'Anorak', 'Tee', 'Tank', 'Top', 'Sweater', 'Flannel', 'Hoodie', 'Cardigan', 'Jacket', 'Henley', 'Poncho', 'Jersey', 'Turtleneck', 'Parka', 'Peacoat', 'Halter', 'Skirt', 'Shorts', 'Jeans', 'Joggers', 'Sweatpants', 'Jeggings', 'Cutoffs', 'Sweatshorts', 'Leggings', 'Culottes', 'Chinos', 'Trunks', 'Sarong', 'Gauchos', 'Jodhpurs', 'Capris', 'Dress', 'Romper', 'Coat', 'Kimono', 'Jumpsuit', 'Robe', 'Caftan', 'Kaftan', 'Coverup', 'Onesie']
data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=150).normalize(imagenet_stats)
learn = create_cnn(data2, models.resnet34);learn.load('stage-1_sz-150')

IMG_FILE_SRC = "static/imageToSave.png"
PREDICTION_FILE_SRC = "static/predictions.txt"

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    fh = open(IMG_FILE_SRC, "wb")
    fh.write(bytes);fh.close()
    return predict_image_from_bytes(bytes)

@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    fh = open(IMG_FILE_SRC, "wb")
    fh.write(bytes);fh.close()
    return predict_image_from_bytes(bytes)

def predict_image_from_bytes(bytes):
    _,_,losses = learn.predict(open_image(BytesIO(bytes)))
    predictions = sorted(zip(classes, map(float, losses)), key=lambda p: p[1], reverse=True)
    fh = open(PREDICTION_FILE_SRC, "w")
    fh.write(str(predictions[0:5]));fh.close()
    return FileResponse('static/result.html')

@app.route("/")
def form(request):
    return FileResponse('static/index.html')

if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8008)
```

This is repo for my article


This web app was created using https://www.starlette.io framework with Fast.AI and is hosted on now.sh in a docker

