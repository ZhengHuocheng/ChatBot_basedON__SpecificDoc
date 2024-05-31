from langchain import OpenAI,PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader

# 确定数据源，需要基于这些数据制作知识库KD
urls = ['https://beebom.com/what-is-nft-explained/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-download-gif-twitter/',
        'https://beebom.com/how-use-chatgpt-linux-terminal/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-save-instagram-story-with-music/',
        'https://beebom.com/how-install-pip-windows/',
        'https://beebom.com/how-check-disk-usage-linux/']

# 加载JS渲染的HTML的URL内容，包含元数据、content和Source(url)

loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()

# 将文本分割，保持上下文联系
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)

# 确定嵌入工具
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
my_activeloop_org_id = "<YOUR-ACTIVELOOP-ORG-ID>"
my_activeloop_dataset_name = "langchain_course_customer_support"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# 将文档上传到deeplake
db.add_documents(docs)

# 进行相似度查询，标准查询都是余弦相似度，可以先使用rerank更能提高retrival的准确性，但是rerank需要额外的token消耗，所以需要权衡。
query = "how to check disk usage in linux?"
# docs = db.similarity_search(query)
# print(docs[0].page_content)

# 定义template

template = """You are an exceptional customer support chatbot that gently answer questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.

Question: {query}

Answer:"""

prompt = PromptTemplate(
    input_variables=["chunks_formatted", "query"],
    template=template,
)

# 检索相关的document chunks
docs = db.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]

# 格式化到template中
chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

# generate answer
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
answer = llm(prompt_formatted)
print(answer)
