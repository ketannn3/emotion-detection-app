import speech_recognition as sr
import pyttsx3
import random

engine = pyttsx3.init()
engine.setProperty('rate',150)


def speak(text):

    print("AI Assistant:", text)

    engine.say(text)
    engine.runAndWait()


def listen():

    r = sr.Recognizer()

    with sr.Microphone() as source:

        print("Speak now...")
        r.adjust_for_ambient_noise(source, duration=2)

        try:
            audio = r.listen(source)

            text = r.recognize_google(audio)

            print("User said:", text)

            return text

        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""

        except Exception as e:
            print("Error:", e)
            return ""


def therapist_reply(emotion):

    responses = {

        "Happy":[
            "You seemed happy during the session. That is wonderful.",
            "Your facial expressions showed happiness. Keep maintaining that positivity."
        ],

        "Sad":[
            "You appeared a little sad during the session. Remember it is okay to feel this way sometimes.",
            "Your expressions suggested sadness. Talking to someone you trust can help."
        ],

        "Angry":[
            "You seemed slightly frustrated. Taking a deep breath might help.",
            "Anger is natural. Try to pause and reflect calmly."
        ],

        "Neutral":[
            "You appeared calm and emotionally balanced during the session.",
            "Your expressions were mostly neutral and composed."
        ],

        "Fear":[
            "Your expressions suggested some anxiety or concern.",
            "You appeared slightly worried during the session."
        ],

        "Surprise":[
            "You showed moments of surprise during the session."
        ],

        "Disgust":[
            "Some expressions suggested discomfort."
        ]
    }

    if emotion in responses:
        return random.choice(responses[emotion])

    return "Emotion detected."


def run_therapist(emotion):

    speak("Hello. I am your AI emotional assistant.")

    reply = therapist_reply(emotion)

    speak(reply)

    print("Listening to user...")

    user_text = listen()

    if user_text != "":

        response = "Thank you for sharing that with me."

        speak(response)

    else:

        speak("I could not hear you clearly.")