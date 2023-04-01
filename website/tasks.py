import io, requests, base64, os, json
from PIL import Image
from .manga_translator import DEFAULT_PARAMATERS
from .models import Chapter, Page, Manga, Box
from . import translator, task_queue, db, scheduler, AppContext, deepl_translator

@scheduler.scheduled_job('interval', seconds=5)
def process_tasks():
    print("Processing Tasks")
    task = task_queue.get()
    task_func, task_args = task
    task_func(*task_args)
    task_queue.task_done()

def TranslateChapter(imgs, name, num, id):
    print("Translating Chapter")
    try:
        with AppContext.get_app().app_context():
            header = {'authorization': f"Bearer {os.getenv('IMGUR_BEARER')}" }
            pages = []
            for img in imgs:
                try:
                    img = Image.open(img)
                    inpainted, region = translator.translate(img, DEFAULT_PARAMATERS)
                    buffer = io.BytesIO()
                    inpainted_bytes = Image.fromarray(inpainted).save(buffer, format="PNG")
                    res = requests.post("https://api.imgur.com/3/image", headers=header, data={ "image": base64.b64encode(buffer.getvalue())}).json()
                    # chapter_page = Page(img_url=res['data']['link'], bounding_box=region)
                    chapter_page = Page(img_url=res['data']['link'])
                    # [([866, 627, 977, 918], '武力行使とは 穏やかではありませんな 魔王リムル よ'), ([772, 717, 839, 980], 'ルベリオスとの戦争を お望みですかな？')]
                    for box in region:
                        chapter_page.bounding_box.append(Box(
                        x=f"{box[0][0]}",
                        y=f"{box[0][1]}",
                        x2=f"{box[0][2]}",
                        y2=f"{box[0][3]}", 
                        jp_text=box[1],
                        translated_en_text=deepl_translator.translate_text(box[1], target_lang='EN-US').text,
                        translated_fr_text=deepl_translator.translate_text(box[1], target_lang='FR').text
                        ))
                    pages.append(chapter_page)
                    
                except Exception as e:
                    print("Error translating Image !", e)
                    return

            print(pages)
            
            selected_manga = Manga.query.filter_by(id=id).first()
            new_chapter = Chapter(name=name, manga_id=id, pages=pages, chapter_index=num)
            selected_manga.chapters.append(new_chapter)
            db.session.commit()
            print("Chapter Translated!")

    except Exception as e:
        print("Exeption during updating! : ", e)



