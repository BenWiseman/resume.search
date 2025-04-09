# Resume Search

This is a simple streamlit app designed to:

1. Read through a collection of uploded resumes
2. Compare each resume against a query (i.e. a job description)
3. Return the top *n* resumes and a brief summary of each

**Notes:**
* No warranty is provided or implied, your hiring decisions are yours at the end of the day.
* It requires an OpenAI API key to run. If running locally it will search for, and default to $OPENAI_API_KEY if present in the environment.
* In an earlier version using free language models the results weren't nearly as good, though you can still fork and modify it to run on USE or sentence_transformers if you like.
