import os
import argparse
import logging
import speech_recognition as sr
from googletrans import Translator
from langdetect import detect
import numpy as np
from hmmlearn import hmm

# List of supported audio file formats
SUPPORTED_FORMATS = ('.m4a', '.mp3', '.webm', '.mp4', '.mpga', '.wav', '.mpeg')


def is_valid_file(file_path):
    """Checks if a file exists and is a regular file."""
    return os.path.isfile(file_path)


def is_supported_format(file_path):
    """Checks if a file is of a supported audio format."""
    return os.path.splitext(file_path)[1].lower() in SUPPORTED_FORMATS


def create_transcript_file(file_path, transcript, confidence_scores=None):
   
    base_filename, _ = os.path.splitext(os.path.basename(file_path))
    output_file_path = f"{base_filename}-transcript.txt"

    if os.path.exists(output_file_path):
        logging.info(f"Output file '{output_file_path}' already exists. Skipping.")
        return

    try:
        # Write transcript and confidence scores (if available) to file
        with open(output_file_path, 'w') as f:
            f.write(transcript)
            if confidence_scores:
                for word, confidence in zip(transcript.split(), confidence_scores):
                    f.write(f"\n  - {word} ({confidence:.2f})")

    except Exception as e:
        logging.error(f"Error while creating transcript file: {e}")


def detect_and_route_language(file_path):
    
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = sr.Recognizer().record(source)
            detected_language = detect(audio_data)
            logging.info(f"Detected language: {detected_language}")

        if detected_language == 'hi':  # Hindi model
            # Cloud-based Speech-to-Text Services (consider alternatives if online access is restricted)
            # Replace with your preferred service's API key/credentials and language code
            from google.cloud import speech_v1p1 as speech
            client = speech.SpeechClient.from_api_key("YOUR_API_KEY")
            audio = speech.Audio(content=audio_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="hi-IN"  # Hindi (India)
            )
            try:
                response = client.recognize(config=config, audio=audio)
                transcript = response.results[0].alternatives[0].transcript
                return transcript
            except Exception as e:
                logging.error(f"Error during Hindi speech recognition: {e}")
                return None

        elif detected_language == 'en':  # English model
            recognizer = sr.Recognizer()
            try:
                with recognizer as source:
                    audio_data = recognizer.record(source)
                    transcript = recognizer.recognize_google(audio_data)
                    return transcript
            except sr.UnknownValueError:
                logging.warning("Speech recognition could not understand audio.")
                return None
            except sr.RequestError as e:
                logging.error(f"Could not request results from Google Speech-to-Text service: {e}")
                return None

                else:
            logging.warning(f"Hinglish or unsupported language detected: {detected_language}")

            recognizer = sr.Recognizer()
            try:
                with recognizer as source:
                    audio_data = recognizer.record(source)
                    transcript = recognizer.recognize_google(audio_data)
                    confidence = recognizer.recognize_google(audio_data, show=False)["confidence"]
                    if confidence < 0.8:  # Adjust threshold based on your needs
                        logging.warning(f"Low confidence ({confidence:.2f}) for recognized transcript in unsupported language: {transcript}")
                    return transcript
            except sr.UnknownValueError:
                logging.warning("Speech recognition could not understand audio.")
                return None
            except sr.RequestError as e:
                logging.error(f"Could not request results from Google Speech-to-Text service: {e}")
                return None
def reduce_noise(audio_data, noise_threshold=0.05):
    """Applies spectral subtraction to the audio data for noise reduction."""
    spectrum = np.fft.fft(audio_data)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # Estimate noise spectrum from silent parts of the audio
    noise_mask = magnitude < noise_threshold * np.max(magnitude)
    noise_spectrum = np.mean(magnitude[noise_mask], axis=0)

    # Apply spectral subtraction
    clean_magnitude = np.maximum(magnitude - noise_spectrum, 0)

    # Reconstruct clean audio signal
    clean_spectrum = clean_magnitude * np.exp(1j * phase)
    clean_audio = np.fft.ifft(clean_spectrum).real.astype(np.int16)

    return clean_audio


def process_directory(directory_path, recursive, translate_to_english=False):
   
    if not os.path.isdir(directory_path):
        logging.error(f"Directory '{directory_path}' does not exist or is not a directory.")
        return

    translator = Translator() if translate_to_english else None  # Initialize translator if needed

    for root, _, files in os.walk(directory_path if recursive else [directory_path]):
        for file in files:
            file_path = os.path.join(root, file)
            if is_supported_format(file_path):
                # Detect language and route to appropriate model
                transcript = detect_and_route_language(file_path)
                if transcript is None:
                    continue

                # Create transcript file
                create_transcript_file(file_path, transcript)

                # Translate transcript to English if requested
                if translate_to_english and translator:
                    try:
                        translated_text = translator.translate(transcript, dest='en').text
                        with open(f"{file_path}-transcript_en.txt", 'w') as f:
                            f.write(translated_text)
                    except Exception as e:
                        logging.error(f"Translation failed: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe and translate audio files.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--file", dest="input_file_path", help="Path to the input audio file.")
    input_group.add_argument("-d", "--directory", dest="input_directory_path", help="Path to the directory containing audio files.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Enable recursion for processing audio files in subdirectories.")
    parser.add_argument("-t", "--translate", action="store_true", help="Translate transcripts to English after recognition.")
    parser.add_argument("-n", "--noise_reduction", action="store_true", help="Apply basic noise reduction (spectral subtraction).")
    return parser.parse_args()


def main():
    """
    Main function that processes command line arguments.
    """
    args = parse_args()

    if args.input_file_path:
        file_path = args.input_file_path
        if not is_valid_file(file_path) or not is_supported_format(file_path):
            logging.error(f"Input file '{file_path}' does not exist, is not a file, or has an unsupported format.")
            return
        process_directory(file_path, args.recursive, args.translate)

    elif args.input_directory_path:
        process_directory(args.input_directory_path, args.recursive, args.translate)

    logging.info("Processing completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
