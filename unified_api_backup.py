
from main import app
from services.facebook_news_assessment import facebook_assess


app.add_api_route(
    "/facebook-assess",
    facebook_assess,
    methods=["POST"],
    tags=["Facebook News"],
    name="facebook_assess"
)
