# Manga MTL Translator

👋 Welcome to the Manga MTL Translator website! This website is designed to translate Japanese manga into others language using machine learning technology.

## Features

🔍 Here are some of the features of our website:

- Machine translation of Japanese manga into all others deepl supported language
- Easy-to-use interface (but ugly)
- The texts are just an overlay so they can be copied and pasted or translated
- Fast and efficient translations
- The images are all uploaded on imgur so nothing is stored locally except the database if it is not explicitly defined

## Installation

🛠️ To install and run this project locally, you'll need to follow these steps:

1. Clone this repository to your local machine with ```git clone https://github.com/Snowad14/Manga-MTL-Website.git```
2. Create a virtual environment using `virtualenv` or `conda`
3. Install the necessary dependencies using `pip install -r requirements.txt` (see pytorch.org to install cuda version)
4. Set the following environment variables in a `.env` file:
   - `IMGUR_BEARER`
   - `UPLOAD_PASSWORD`
   - `APP_SECRET_KEY`
   - `DEEPL_KEY`
   - `SQLALCHEMY_DATABASE_URI` (optional but necessary if you want an online database)
5. Run the app using `python main.py`

## Usage

🖥️ To use the Manga MTL Translator website, follow these simple steps:

1. Upload a manga chapter to the "Add" section of the website
2. Click the "Add Chapter" button by choosing a manga previously added
3. Enjoy your translated manga!

## Screenshots

![Discover section](https://i.imgur.com/Utf6bef.png)
![Reader](https://i.imgur.com/yakwB5l.png)