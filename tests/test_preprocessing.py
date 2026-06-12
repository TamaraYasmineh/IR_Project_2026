from services.preprocessing_service import preprocess_text


sample_text = """
What is the step by step guide to invest in share market in India?
Visit https://google.com
Email me at test@gmail.com
I have 100 dollars.
"""

processed_text = preprocess_text(sample_text)

print("Original Text:")
print(sample_text)

print("\nProcessed Tokens:")
print(processed_text)