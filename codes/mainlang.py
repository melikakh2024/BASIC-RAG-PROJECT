
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings


from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode
from qdrant_client import QdrantClient, models


from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


loader = PyPDFLoader(
file_path = "./langchain_rag_sample_4_pages.pdf",
# headers = None
# password = None,
mode = "page",
pages_delimiter = ()
)

docs=loader.load()
# print(len(docs))
# print(docs)
# print(type(docs[0]))

# print(len(docs[0].page_content))

# text=docs[0].page_content
# cnt=print(len(text.strip().split()))

docp=RecursiveCharacterTextSplitter( chunk_size=500,chunk_overlap= 200)
p=docp.split_documents(docs)
# print(type(p))
# print(len(p))
# print(p[0])
# print(p[20])
# print(p[0])
# text=p[0].page_content
# print(len(text.split()))
# print(len(p[0].page_content))
# print(len(p[20].page_content))

embed = OllamaEmbeddings(model= "nomic-embed-text")

# # # input_texts=[i.page_content for i in p]
# # input_texts=[]
# # final=[]

# # for i in p:
# #     vector = embed.embed_documents([i.page_content])
# #     dict={}
# #     dict['vectors']=vector
# #     dict['metadata']=i.metadata
# #     dict['content']=i.page_content
# #     final.append(dict)

client = QdrantClient(url="http://localhost:6333")

if not client.collection_exists(collection_name="docs"):
    qdrant = QdrantVectorStore.from_documents(
        documents=p,
        embedding=embed,
        collection_name="docs",
        url="http://localhost:6333",
    )
else:
    qdrant = QdrantVectorStore.from_existing_collection(
        collection_name="docs",
        embedding=embed,
        url="http://localhost:6333",
    )

query = input("Enter user's question:")
found_docs = qdrant.similarity_search(query)

# for j in found_docs:
#     print(j.page_content)

prompt_template = ChatPromptTemplate.from_template("""
Answer the user's question based only on the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
""")

context_text = "\n\n".join([doc.page_content for doc in found_docs])

final_prompt = prompt_template.format(
    context=context_text,
    question=query
)

llm = ChatOllama(model="llama3.2")
response = llm.invoke(final_prompt)

print("\n--- Answer ---")
print(response.content)