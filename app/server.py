from starlette.applications import Starlette
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
from starlette.routing import Router, Mount
import uvicorn, aiohttp, asyncio
from io import BytesIO
from fastai import *
from fastai.vision import *
import base64

# model_file_url = 'https://github.com/pankymathur/cloth_finder_app/blob/master/data/cloth_categories/models/stage-1_sz-150.pth'
# model_file_name = 'model'
# classes = ['Blouse', 'Blazer', 'Button-Down', 'Bomber', 'Anorak', 'Tee', 'Tank', 'Top', 'Sweater', 'Flannel', 'Hoodie', 'Cardigan', 'Jacket', 'Henley', 'Poncho', 'Jersey', 'Turtleneck', 'Parka', 'Peacoat', 'Halter', 'Skirt', 'Shorts', 'Jeans', 'Joggers', 'Sweatpants', 'Jeggings', 'Cutoffs', 'Sweatshorts', 'Leggings', 'Culottes', 'Chinos', 'Trunks', 'Sarong', 'Gauchos', 'Jodhpurs', 'Capris', 'Dress', 'Romper', 'Coat', 'Kimono', 'Jumpsuit', 'Robe', 'Caftan', 'Kaftan', 'Coverup', 'Onesie']

model_file_url = 'https://www.dropbox.com/s/y4kl2gv1akv7y4i/stage-2.pth?raw=1'
model_file_name = 'model'
classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

app = Router(routes=[Mount('/static', app=StaticFiles(directory='static')),])

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        tfms=get_transforms(), size=150).normalize(imagenet_stats)
    learn = create_cnn(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

IMG_FILE_SRC = "static/saved_image.png"
PREDICTION_FILE_SRC = "static/predictions.txt"

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["img"].read())
    bytes = base64.b64decode(img_bytes)
    return predict_from_bytes(bytes)

def predict_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _,_,losses = learn.predict(img)
    img.save(IMG_FILE_SRC)
    predictions = sorted(zip(classes, map(float, losses)), key=lambda p: p[1], reverse=True)
    fh = open(PREDICTION_FILE_SRC, "w")
    fh.write(str(predictions[0:3]));fh.close()
    return FileResponse('static/result.html')

@app.route("/")
def form(request):
    return FileResponse('static/index.html')

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8008)