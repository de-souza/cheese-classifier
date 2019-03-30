import os
from io import BytesIO
import aiohttp
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.responses import HTMLResponse
from fastai.vision import load_learner, open_image
from torch.nn.functional import softmax


app = Starlette()
app.add_middleware(HTTPSRedirectMiddleware)

learn = load_learner("data")


@app.route("/")
def form(request):
    return HTMLResponse(
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        '  <meta charset="utf-8">\n'
        "  <title>Cheese Classifier</title>\n"
        "  <h1>Cheese Classifier</h1>\n"
        "  <p>Select image to upload:\n"
        '  <form action="/upload" method="post" enctype="multipart/form-data">\n'
        '    <input type="file" name="file">\n'
        '    <input type="submit" value="Upload Image">\n'
        "  </form>\n"
        "  <p>Or submit a URL:\n"
        '  <form action="/classify-url" method="get">\n'
        '    <input type="url" name="url">\n'
        '    <input type="submit" value="Fetch and analyze image">\n'
        "  </form>"
    )


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await data["file"].read()
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    prediction, _, outputs = learn.predict(img)
    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        '  <meta charset="utf-8">\n'
        "  <title>Cheese Classifier</title>\n"
        "  <h1>Cheese Classifier</h1>\n"
        f"  <p>Prediction: <strong>{prediction}</strong>\n"
        "  <p>Output:\n"
        "  <table>\n"
    )
    for pred_class, pred_prob in zip(learn.data.classes, softmax(outputs, dim=0)):
        html += (
            "    <tr>\n"
            f"      <th>{pred_class}\n"
            f"      <td>{pred_prob:.1%}\n"
        )
    html += "  </table>"
    return HTMLResponse(html)


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
