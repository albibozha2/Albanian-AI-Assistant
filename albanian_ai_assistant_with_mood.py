import cv2
import face_recognition
import pyttsx3
import speech_recognition as sr
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorflow.keras.models import load_model
import numpy as np

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# Load a model and tokenizer for text generation (Albanian)
model_name = "flax-community/gpt2-small-albanian"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize speech recognition
recognizer = sr.Recognizer()

# Load the facial expression recognition model
emotion_model = load_model('emotion_detection_model.hdf5')

# Emotion labels corresponding to the model's output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print(colored("Po pres zërin tuaj...", 'green'))
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="sq-AL")
            print(colored(f"Ju thatë: {text}", 'blue'))
            return text
        except sr.UnknownValueError:
            print(colored("Nuk kuptova, ju lutem provoni përsëri.", 'red'))
            return None
        except sr.RequestError as e:
            print(colored(f"Shërbimi i njohjes së zërit nuk është i disponueshëm; {e}", 'red'))
            return None

def generate_response(prompt, mood):
    # Customize responses based on mood
    mood_prefix = {
        'Happy': 'Duket se jeni i lumtur! ',
        'Sad': 'Më vjen keq që jeni i trishtuar. ',
        'Angry': 'Duket se jeni i inatosur. ',
        'Surprise': 'Sa e papritur! ',
        'Neutral': 'Më duket se jeni neutral. '
    }
    
    # Prefix the mood to the prompt
    prompt = mood_prefix.get(mood, '') + prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def detect_mood(frame):
    # Convert frame to grayscale for emotion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_recognition.face_locations(gray_frame)

    for (top, right, bottom, left) in faces:
        face = gray_frame[top:bottom, left:right]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Predict emotion
        emotion_prediction = emotion_model.predict(face)
        max_index = np.argmax(emotion_prediction[0])
        emotion = emotion_labels[max_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        return emotion

    return 'Neutral'

def face_tracking_and_mood_detection():
    video_capture = cv2.VideoCapture(0)
    detected_mood = 'Neutral'

    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect mood
        detected_mood = detect_mood(rgb_frame)

        # Display the frame
        cv2.imshow('Face Tracking and Mood Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return detected_mood

def run_ai_assistant():
    print(colored("AI Asistenti Shqiptar është gati! Shtyp 'exit' për të dalë.", 'green'))
    speak("AI Asistenti Shqiptar është gati! Shtyp exit për të dalë.")
    
    while True:
        prompt = listen()

        if prompt is None:
            continue

        if prompt.lower() in ['exit', 'dalje', 'mbyll']:
            speak("Mbyllja e AI Asistentit.")
            print(colored("Mbyllja e AI Asistentit...", 'red'))
            break

        if 'fytyra' in prompt.lower():
            speak("Po filloj ndjekjen e fytyrës dhe zbulimin e gjendjes emocionale.")
            detected_mood = face_tracking_and_mood_detection()
            speak(f"Kam zbuluar që jeni në gjendje {detected_mood}.")
            continue

        detected_mood = face_tracking_and_mood_detection()
        response = generate_response(prompt, detected_mood)
        print(colored(f"AI: {response}", 'green'))
        speak(response)

if __name__ == "__main__":
    run_ai_assistant()