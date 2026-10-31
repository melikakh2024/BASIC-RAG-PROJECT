from pathlib import Path
import ollama
from qdrant_client import QdrantClient



doc={}
path = Path('./Documents')
for i in path.glob('*.txt'):
    with open (i,"r") as f:
        key=i.stem
        all_txt=f.read()
        if all_txt.strip()=="":
            continue
        doc[key]=all_txt


doc_list=[{"filename":key,"content":value} for key,value in doc.items()]

chunk=8
overlap=3
step=chunk-overlap

final=[]
for i in range(0,len(doc_list)):
    words=doc_list[i]['content'].split()

    for j in range(0,len(words),step):
        A={}
        A["filename"]= doc_list[i]["filename"]
        A["content"]= " ".join(words[j:j+chunk])
        final.append(A)
# print(final)

new=[]
for i in final:
    response = ollama.embed(
        model="nomic-embed-text",
        input=i["content"])
    B={}
    B['filename']=i['filename']
    B['vector']=response.embeddings[0]
    B['content']=i['content']

    new.append(B)
# print(len(final))
# print(len(new))
# print(new[0].keys())

# # print(len(response.embeddings[0]))
# print(len(response.embeddings))
# print(response.embeddings)
client = QdrantClient(url="http://localhost:6333")

# print(client.get_collections())


from qdrant_client.models import Distance, VectorParams

if not client.collection_exists("documents"):
    client.create_collection(
        collection_name="documents",
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )
# print(client.get_collections())


from qdrant_client.models import PointStruct

points = [
    PointStruct(
        id=idx,
        vector=item["vector"],
        payload={
            "filename": item["filename"],
            "content": item["content"]
        }
    )
    for idx, item in enumerate(new)
]

# print(len(points))
# print(points[0].vector[0:20])
# print(points[0].id)
# print(points[0].payload)

client.upsert(
    collection_name="documents",
    points=points
)

result = client.count(
    collection_name="documents",
    exact=True
)

# print(result.count)

question = input("Ask your question: ")

resp=ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_vector = resp.embeddings[0]

results = client.query_points(
    collection_name="documents",
    query=question_vector,
)

context = ""

for point in results.points:
    context += point.payload["content"] + "\n"

# print(context)
prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

response = ollama.generate(
    model="llama3.2",
    prompt=prompt,
)

print(response.response)