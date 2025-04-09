import os
import pickle
import faiss
import numpy as np
from typing import List, Tuple, Optional

# We'll re-use embed_text_openai from your existing semantic_py.py:
from embed import embed_text_openai  

# For advanced usage, you might also import embed_text_use or other embedding funcs.


class VectorDB:
    """
    Simple FAISS-based vector database that can be saved/loaded from disk.
    Stores:
      - self.index: a FAISS index containing vectors
      - self.documents: list of dicts, each with {"path": str, "text": str}
    """

    def __init__(self, dimension: int):
        """
        Initialize a new empty FAISS index with a given dimension.
        dimension: Embedding dimension.
        """
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []  # parallel data structure to store text + file paths

    def add_documents(self, embeddings: np.ndarray, doc_info: List[dict]):
        """
        Add new documents and their embeddings to the database.

        embeddings: shape (N, dimension)
        doc_info: list of dicts, each with {"path": str, "text": str}
                  length must be N.
        """
        if embeddings.shape[0] != len(doc_info):
            raise ValueError("Mismatch between number of embeddings and doc_info items.")

        self.index.add(embeddings)
        self.documents.extend(doc_info)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        Search for the top_k documents most similar to the query_embedding.

        query_embedding: shape (1, dimension) or (dimension,) for a single query.
        Returns: A list of dicts: [{"path": str, "text": str, "distance": float}, ...]
        """
        # Ensure correct shape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            doc_dict = {
                "file": self.documents[idx]["file"],
                "text": self.documents[idx]["text"],
                "distance": float(dist),
            }
            results.append(doc_dict)
        return results

    def save_to_disk(self, index_path: str, docs_path: str):
        """
        Save the FAISS index to 'index_path' and the documents list to 'docs_path'.
        """
        faiss.write_index(self.index, index_path)
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load_from_disk(cls, index_path: str, docs_path: str):
        """
        Load a VectorDB from a previously saved FAISS index and documents.
        """
        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)

        # Create an instance. We must ensure the index dimension matches the embeddings
        dim = index.d
        db = cls(dimension=dim)
        db.index = index
        db.documents = docs
        return db


def embed_directory_of_resumes(
    directory_path: str,
    embed_func=embed_text_openai,
    model: str = "text-embedding-3-large"
) -> Tuple[np.ndarray, List[dict]]:
    """
    Reads each text file in `directory_path`, embeds it, 
    and returns a tuple: (embeddings [NumPy], doc_info [list of dicts]).
    doc_info[i] has {"path": path, "text": text_of_file}.
    """
    doc_info = []
    file_list = [f for f in os.listdir(directory_path) if f.lower().endswith(".txt")]

    if not file_list:
        raise ValueError("No .txt files found in the specified directory.")

    for filename in file_list:
        file_path = os.path.join(directory_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        doc_info.append({"file": file_path, "text": text})

    # Embed all texts
    texts = [d["text"] for d in doc_info]
    embeddings_list = embed_func(texts, model=model)
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    return embeddings_array, doc_info


def search_resumes(
    job_description: str,
    db: VectorDB,
    embed_func=embed_text_openai,
    model: str = "text-embedding-3-large",
    top_k: int = 5
) -> List[dict]:
    """
    Given a job description (or user query), embed it, then search the VectorDB for top_k similar resumes.
    Returns a list of dicts: [{"path": ..., "text": ..., "distance": ...}, ...]
    """
    query_embed_list = embed_func([job_description], model=model)
    query_embed = np.array(query_embed_list, dtype=np.float32)
    results = db.search(query_embed, top_k=top_k)
    return results


def summarize_resume(
    resume_text: str,
    query: str = "no specific query is needed",
    model: str = "gpt-4o",  # default per your example
    api_key: str = None,
    tokens: int = 500
) -> str:
    """
    Summarize a single resume's text using the new v1/chat/completions endpoint.

    :param resume_text: The full text of the resume.
    :param query: The user query or job description.
    :param model: The Chat Completion model to use (e.g. 'gpt-4', 'gpt-4o', 'gpt-3.5-turbo').
    :param api_key: Optional API key. If not supplied, the function will look in the environment.
    :param tokens: max tokens to use.
    :return: A concise summary as a string.
    """
    from openai import OpenAI

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set and no api_key was provided.")

    client = OpenAI(api_key=api_key)

    # Use the Chat Completion endpoint
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Please summarize the following resume in at most one paragraph:\n\n"
                    f"{resume_text}"
                    "Make the summary relevant to the supplied query (which may be a simple request or a job description.):\n\n"
                    f"{query}"
                )
            }
        ],
        max_tokens=tokens  # Adjust as needed
    )

    # The returned object is a pydantic model
    # The actual summary is in completion.choices[0].message.content
    summary_text = completion.choices[0].message.content.strip()
    return summary_text
  
  
