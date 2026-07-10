from transformers import pipeline, AutoFeatureExtractor

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")

if __name__ == "__main__":
    result = summarizer("Long article text here...", max_length=130)
    print(result)
