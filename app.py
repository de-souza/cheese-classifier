import os
from io import BytesIO
import aiohttp
import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from fastai.vision import load_learner, open_image
from torch.nn.functional import softmax


app = Starlette()

learn = load_learner(".")


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="en-gb">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Cheese Classifier</title>
          </head>
          <body>
            <h1>Cheese Classifier</h1>
            <p>Select an image to upload:</p>
            <form action="/upload" method="post" enctype="multipart/form-data">
              <input type="file" name="file">
              <input type="submit" value="Upload Image">
            </form>
            <p>Or submit a URL:</p>
            <form action="/classify-url" method="get">
              <input type="url" name="url">
              <input type="submit" value="Fetch and analyze image">
            </form>
          </body>
        </html>
        """
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
    pred_class, _, outputs = learn.predict(img)
    formatted_outputs = [f"{x:.1%}" for x in softmax(outputs, dim=0)]
    pred_probs = sorted(
        zip(learn.data.classes, formatted_outputs),
        key=lambda p: p[1],
        reverse=True,
    )
    return HTMLResponse(
        f"""
        <!DOCTYPE html>
        <html lang="en-gb">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Cheese Classifier</title>
          </head>
          <body>
            <h1>Cheese Classifier</h1>
            <p>Guess: {pred_class}</p>
            <p>Results: {pred_probs}</p>
          </body>
        </html>
        """
    )


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
