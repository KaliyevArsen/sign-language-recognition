# Program by Kaliyev.A
import telebot
from requests.exceptions import ReadTimeout
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import numpy as np
import io

TOKEN = 'BOT TOKEN'
bot = telebot.TeleBot(TOKEN)

model_path = 'model.keras'
model = load_model(model_path)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Скинь мне фотографию буквы на английском языке жестов, и я ее угадаю!")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        image_stream = io.BytesIO(downloaded_file)
        image_stream.seek(0)
        image = Image.open(image_stream).convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        # prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)

        response = f"Это буква '{predicted_class}'!"
        bot.reply_to(message, response)

    except Exception as e:
        bot.reply_to(message, f"Ошибка в бэкенде: {e}")


bot.polling(none_stop=True, interval=1)
