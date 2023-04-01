import os, io
from flask import Blueprint, render_template, request, flash, redirect, url_for
from .tasks import TranslateChapter
from .models import Manga
from . import db, task_queue

new = Blueprint("new", __name__)

@new.route("/create_manga", methods=['POST'])
def create_manga():
    if request.form.get('name') and request.form.get('author') and request.form.get('description') and request.form.get('cover'):
        manga = Manga(name=request.form['name'], author=request.form['author'], description=request.form['description'], cover=request.form['cover'])
        db.session.add(manga)
        db.session.commit()
        flash('Manga created!', category="success")
        return redirect(url_for('views.homeMenu'))
    else:
        flash('Please fill in all fields', category="error")
        return redirect(url_for('views.NewDataPage'))

@new.route("/add_chapter", methods=['POST'])
def addMangaChapter():
    
    error_messages = []
    if not request.form.get('title'):
        error_messages.append('Please enter a chapter title')
    if len([img for img in request.files.getlist('imgs') if img.filename]) == 0:
        error_messages.append('Please upload at least one image')
    
    for message in error_messages:
        flash(message, category='error')
        return redirect(url_for('views.NewDataPage'))
        
    uploaded_images = [io.BytesIO(image.read()) for image in request.files.getlist('imgs')]
    task_queue.put((TranslateChapter, (uploaded_images, request.form.get('title'), request.form.get('ChapterNumber'), request.form.get('manga_id'))))
    flash('Chapter will be translated and added!', category="success")
    return redirect(url_for('views.homeMenu'))



