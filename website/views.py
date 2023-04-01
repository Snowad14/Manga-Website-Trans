from flask import Blueprint, render_template, request, jsonify
from .models import Manga

views = Blueprint("views", __name__)

@views.route("/")
def homeMenu():
    mangas = Manga.query.limit(3).all()
    return render_template('discover.html', mangas=mangas) 

@views.route("/search")
def searchPage():
    return render_template('search.html') 

@views.route("/new")
def NewDataPage():
    mangas = Manga.query.all()
    return render_template('new.html', mangas=mangas)

@views.route("/manga/<int:manga_id>" , methods=['GET'])
def showMangaChapters(manga_id):
    manga = Manga.query.filter_by(id=manga_id).first()
    chapters = manga.chapters.all()
    if manga:
        return render_template('chapters.html', manga=manga, chapters=chapters)
    else:
        return jsonify(error="Manga ID not found"), 404

@views.route("/reader/<int:manga_id>/<int:chapter_id>" , methods=['GET'])
def showChapterPages(manga_id, chapter_id):
    manga = Manga.query.filter_by(id=manga_id).first()
    if manga:
        chapter = manga.chapters.filter_by(id=chapter_id).first()
        if chapter:
            pages = chapter.pages.all()
            return render_template('reader.html', manga=manga, chapter=chapter, pages=pages)
        else:
            return jsonify(error="Chapter not found"), 404
    else:
        return jsonify(error="Manga ID not found"), 404
