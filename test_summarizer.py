from app.services.summarizer import VideoSummarizer


test_text = """
The former head of the State Intelligence Service
has been admitted to the Colombo National Hospital
following an illness.

He is currently being held by the Criminal
Investigation Department under a detention order.

He remains under interrogation regarding alleged
links to the Easter Sunday attacks.
"""


print("\nStarting summarization test...\n")


summarizer = VideoSummarizer()


result = summarizer.summarize(
    transcript=test_text,
    language="en"
)


print("\n--- SUMMARY RESULT ---\n")

print("Language:")
print(result["language"])

print("\nSummary:")
print(result["summary"])

print("\nChunks:")
print(result["chunk_count"])

print("\nModel:")
print(result["model"])