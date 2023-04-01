from dotenv import load_dotenv; load_dotenv()
from website import create_app, AppContext

app = create_app()

if __name__ == "__main__":
    AppContext.set_app(app)
    app.run(debug=True, use_reloader=True)