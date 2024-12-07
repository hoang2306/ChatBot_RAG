from langchain_ollama import OllamaLLM
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import sentence_transformers
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import gradio as gr
import os 
from pyvi.ViTokenizer import tokenize
from functools import lru_cache
import torch
import numpy as np 
torch.cuda.empty_cache()

from env import (
    MUSIC_DATA_FOLDER, 
    DISEASE_DATA_FOLDER, 
    API_KEY, 
    CLOUD_ID,
    INDEX_NAME
)

from parser import get_cmd

class GenerateTextCallback(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token, **kwargs) -> None:
        print(token, end='', flush=True)

# # Tạo callback handler
gen_text_callback_handler = GenerateTextCallback()
# llm = OllamaLLM(
#     model='phi3:3.8b',
#     format='',
#     callbacks=[gen_text_callback_handler]
# )

# output_parser = StrOutputParser()

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant. Please respond to the user's queries."),
#         ("user", "Question: {question}")
#     ]
# )

# chain = prompt | llm | output_parser

# # Hàm chính để stream output từ LLM
# def generate_response_stream(question):
#     # Reset trạng thái token trước mỗi yêu cầu mới
#     gen_text_callback_handler.tokens.clear()
    
#     # Chạy chain và gửi từng token qua generator
#     chain.invoke(question)
#     for token in gen_text_callback_handler.tokens:
#         yield "".join(gen_text_callback_handler.tokens)  # Gửi kết quả từng phần


# # get docs 
# from urllib.request import urlopen
# import os, json

# url = "https://raw.githubusercontent.com/ashishtiwari1993/langchain-elasticsearch-RAG/main/data.json"

# response = urlopen(url)

# workplace_docs = json.loads(response.read())

# # pre-process docs
# metadata = []
# content = []

# for doc in workplace_docs:
#   content.append(doc["content"])
#   metadata.append({
#       "name": doc["name"],
#       "summary": doc["summary"],
#       "rolePermissions":doc["rolePermissions"]
#   })

# text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
# docs = text_splitter.create_documents(content, metadatas=metadata)

# # get embedding 
# print('starting load embeddings from HuggingFace .....')
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
# print('done load embeddings from HuggingFace')

# # env variable


# print('starting init Elastic Search')
# es = ElasticsearchStore.from_documents(
#     docs, 
#     es_cloud_id=CLOUD_ID,
#     index_name=INDEX_NAME,
#     es_api_key=API_KEY,
#     embedding=embeddings
# )
# print('done init es')


# model for similarity 
# can use model in embedding 
model_sim = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')

# sample question 
disease_related_samples = [
    "What are the symptoms of cardiovascular disease?",
    "How can I prevent diabetes?",
    "What treatments are available for cancer?"
]

# Encode the disease-related questions
disease_embeddings = model_sim.encode(
    sentences=disease_related_samples, 
    convert_to_tensor=True
)

print(f'disease_embedding shape: {disease_embeddings.shape}')


def preprocess_doc(file_path, chunk_size=500, chunk_overlap=0):
    loader = TextLoader(file_path=file_path, encoding='utf-8')
    docs = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size, 
    #     chunk_overlap=chunk_overlap
    # )
    # result = text_splitter.split_documents(docs)

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    result = text_splitter.split_documents(docs)

    return result

"""
- problem: dùng thư viên trực tiếp như huggingfaceembeddings nó chỉ load với inference model chứ ko tokenize text bằng pyvi giống như yêu cầu của model tiếng Việt

- solution: kế thừa class HuggingFaceEmbeddings lại rồi overide cái method embed documents của nó

https://github.com/ausarhuy/rag-agent/blob/master/src/embeddings.py?fbclid=IwY2xjawHBt0xleHRuA2FlbQIxMAABHeVkn6s4Z8oMWuX-CrTfw_yPjFlq_9skduvAPYtjwJ6AQfOdQNo5TkkqHA_aem_ixLBxoTWpuWSsdPOnuQu3g
"""
class VietnameseEmbeddings(HuggingFaceEmbeddings):

    def embed_documents(self, texts):
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(tokenize, texts))
        if self.multi_process:
            pool = self._client.start_multi_process_pool()
            embeddings = self._client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self._client.encode(
                texts, show_progress_bar=self.show_progress, **self.encode_kwargs
            )

        return embeddings.tolist()


@lru_cache()
def get_embedding_model():
    return VietnameseEmbeddings(model_name='dangvantuan/vietnamese-document-embedding',
                                model_kwargs={
                                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                                    'trust_remote_code': True
                                    })


# embedding_function = get_embedding_model()

def get_retriever(conf, data_path):
    # get embedding from huggingface
    print('starting load embeddings from HuggingFace .....')
    model_kwargs = {'device': 'cpu'}

    if conf['domain'] == 0:
        embeddings = get_embedding_model()
    else: 
        embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-small-en',
            model_kwargs=model_kwargs
        )

    print('done load embeddings from HuggingFace')

    print('starting load data ....')

    print(f'list dir: {os.listdir(data_path)}')
    all_splits = []
    for dirpath, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.endswith(".txt"):
                filepath = os.path.join(dirpath, filename)
                print(f"Processing file: {filepath}")
                splits = preprocess_doc(filepath)
                all_splits.extend(splits)

    print(f'all_split: {all_splits}')
    print('done load data !!!')

    # elastic search 
    print('starting init Elastic Search')
    # es = ElasticsearchStore.from_documents(
    #     all_splits, 
    #     es_cloud_id=CLOUD_ID,
    #     index_name=INDEX_NAME,
    #     es_api_key=API_KEY,
    #     embedding=embeddings
    # )
    es = ElasticsearchStore(
        es_cloud_id=CLOUD_ID,
        index_name=INDEX_NAME,
        es_api_key=API_KEY,
        embedding=embeddings
    )
    print('done init es')

    # vectorstore = FAISS.from_documents(
    #     documents=all_splits, 
    #     embedding=embeddings
    # )

    # return vectorstore.as_retriever(
    #     search_type="similarity", 
    #     search_kwargs={"k": 3} # knn 
    # )

    return es

def check_related(query):
    q_emb = model_sim.encode(query, convert_to_tensor=True)

    score = [
        cosine_similarity(q_emb.reshape(-1,1), t_emb.reshape(-1,1)) for t_emb in disease_embeddings
    ]

    return np.max(score) >= 0.5


def get_rag(conf):
    # fixed path
    data_path = MUSIC_DATA_FOLDER if conf['domain'] == 0 else DISEASE_DATA_FOLDER
    retriever = get_retriever(conf=conf, data_path=data_path)
    model_list = {
        1: 'llama2',
        2: 'phi3:3.8b',
        3: 'mrjacktung/phogpt-4b-chat-gguf',
        4: 'Tuanpham/t-visstar-7b:latest'
    }
    llm = OllamaLLM(
        model=model_list[conf['model_ollama']],
        callbacks=[gen_text_callback_handler]
    )

    
    
    # domain=0: music, domain=1: disease
    if conf['domain']:
        prompt = """
        Bạn là một chuyên gia âm nhạc giúp trả lời các câu hỏi dựa trên ngữ cảnh được cung cấp. Ngữ cảnh là thông tin về các ban nhạc rock, lịch sử của họ, các album và buổi biểu diễn.
        Hãy trả lời câu hỏi dưới đây dựa trên ngữ cảnh. Nếu bạn không biết câu trả lời, chỉ cần nói 'Tôi không biết'.

        Ngữ cảnh:
        {context}

        Câu hỏi: {question}
        Câu trả lời:
        """
    else:
        prompt = """
        You are a doctor that helps to answer questions based on the context provided. The context is information from documents related to healthcare topics.
        Answer the following question based on the context below. If you do not know the answer, simply say 'I don't know'.

        Context:
        {context}

        Question: {question}
        Answer:
        """


    def retrieve_from_db(query):
        # docs = retriever.invoke(query)

        # elastic search
        docs = retriever.similarity_search(query=query)
    
        # format docs 
        result = "\n\n".join(doc.page_content for doc in docs)
        return result

    def rag(query):
        # if not check_related(query=query):
        #     answer = llm.invoke()
        #     return answer
        
        context = retrieve_from_db(query)
        print(f'query: {query}')
        print(f'context: {context}')
        prompt_RAG = prompt.format(
            context=context,
            question=query
        )
        result = llm.invoke(prompt_RAG)
        return result
        print('-'*60)
    
    return rag 

def process_question(question, history=None):
    context = ""

    context += f"\nUser: {question}\n"
    response = rag(question)

    return response


# parser 
args = get_cmd().__dict__
print(f'args list: {args}')

# create rag chain
print('creating RAG ....') 
rag = get_rag(conf=args)
print('done create RAG')

# disease: dữ liệu tiếng anh
# music: dữ liệu tiếng việt

if args['domain'] == 0:
    examples_music = [
        "lead guitarist ban nhạc ngũ cung là ai?",
        "album gần đây nhất của ban nhạc bức tường",
        "tên thành viên các ban nhạc ngọt"
    ]
else: 
    examples_disease = [
        "What are different kinds of diseases?",
        "How to prevent heart disease?",
        "Symptoms of diabetes"
    ]


# deploy on gradio
gr.ChatInterface(
        fn=process_question,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Ask me a question", container=False),
        title="Healthcare Chatbot",
        examples=examples_music,
    ).launch(share=args['public'])



