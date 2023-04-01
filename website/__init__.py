import os, queue, deepl
import pymysql; pymysql.install_as_MySQLdb()
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from apscheduler.schedulers.background import BackgroundScheduler
from .manga_translator import MangaTranslator, DEFAULT_PARAMATERS

db = SQLAlchemy()
translator = MangaTranslator(DEFAULT_PARAMATERS)
deepl_translator = deepl.Translator(os.getenv("DEEPL_KEY"))
task_queue = queue.Queue()
scheduler = BackgroundScheduler()
DB_NAME = "database.db"

class AppContext: # I know this is not good, but others alternatives like current_app are not working
    app = None
    
    @staticmethod
    def set_app(app):
        AppContext.app = app
    
    @staticmethod
    def get_app():
        return AppContext.app

def create_app():

    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("APP_SECRET_KEY")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DB_URI") if os.getenv("DB_URI") else f"sqlite:///{DB_NAME}"
    db.init_app(app)

    from .views import views
    app.register_blueprint(views, url_prefix="/")

    from .creator import new
    app.register_blueprint(new, url_prefix="/")

    from .models import Manga
    create_database(app)

    scheduler.start()
    return app

def create_database(app):
    if not os.path.exists(f"website/{DB_NAME}"):
        with app.app_context():
            db.create_all()
            print("Created Database!")
