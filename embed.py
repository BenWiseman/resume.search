import os

'''
try:
    import tensorflow_hub as hub
except ImportError:
    raise ImportError("tensorflow_hub is not installed. Install it if you want to use the USE embedder.")
'''
# If you prefer the global openai instance instead of an instantiated client:
#     import openai
#     openai.api_key = os.environ.get("OPENAI_API_KEY")  # or handle errors
# Then embed with: 
#     openai.embeddings.create(model="text-embedding-ada-002", input=["some text"])

# But below shows the recommended approach with the new OpenAI client:

def get_openai_api_key():
    """
    Retrieve the OpenAI API key from the OS environment.
    Raise an error if not found.
    """
    key = os.environ.get("OPENAI_API_KEY", None)
    if not key:
        raise ValueError("OPENAI_API_KEY is not set in the OS environment.")
    return key

def embed_text_openai(texts, model="text-embedding-3-large", api_key=None):
    """
    Create embeddings with the openai>=1.0.0 library by instantiating a client.
    `texts` should be a list of strings.
    """
    from openai import OpenAI

    if api_key is None:
        api_key = get_openai_api_key()

    # Instantiate the client
    client = OpenAI(api_key=api_key)

    # The new openai client expects a list for `input`
    # The response is a pydantic model, not a dict
    response = client.embeddings.create(model=model, input=texts)

    # Each embedding is in response.data[i].embedding
    # We'll extract them all
    embeddings = [record.embedding for record in response.data]
    return embeddings

'''
def embed_text_use(texts, model_url="https://tfhub.dev/google/universal-sentence-encoder/4"):
    """
    Create embeddings via Google Universal Sentence Encoder.
    """
    embed = hub.load(model_url)
    embeddings = embed(texts)
    return embeddings.numpy().tolist()
'''