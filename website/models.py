from . import db
from sqlalchemy.sql import func

# I didn't manage to call an integer with javascript because it gives me all in python byte format so the coordinates are in string
class Box(db.Model):
    __tablename__ = 'box'
    id = db.Column(db.Integer, primary_key=True)
    x = db.Column(db.String(32))
    y = db.Column(db.String(32))
    x2 = db.Column(db.String(32))
    y2 = db.Column(db.String(32))
    page_id = db.Column(db.Integer, db.ForeignKey('page.id'))
    jp_text = db.Column(db.String(1024))
    translated_en_text = db.Column(db.String(1024))
    translated_fr_text = db.Column(db.String(1024))

    def __repr__(self):
        return '<Box %r>' % self.id

class Page(db.Model):
    __tablename__ = 'page'
    id = db.Column(db.Integer, primary_key=True)
    chapter_id = db.Column(db.Integer, db.ForeignKey('chapter.id'))
    img_url = db.Column(db.String(64))
    bounding_box = db.relationship('Box', backref='Page', lazy='dynamic')

    def __repr__(self):
        return '<Page %r>' % self.id

class Chapter(db.Model):
    __tablename__ = 'chapter'
    id = db.Column(db.Integer, primary_key=True)
    chapter_index = db.Column(db.Integer)
    name = db.Column(db.String(64), unique=True, index=True)
    manga_id = db.Column(db.Integer, db.ForeignKey('manga.id'))
    pages = db.relationship('Page', backref='chapter', lazy='dynamic')

    def __repr__(self):
        return '<Chapter %r>' % self.name

class Manga(db.Model):
    __tablename__ = 'manga'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, index=True)
    author = db.Column(db.String(64))
    description = db.Column(db.Text)
    cover = db.Column(db.String(64))
    chapters = db.relationship('Chapter', backref='manga', lazy='dynamic')

    def __repr__(self):
        return '<Manga %r>' % self.name