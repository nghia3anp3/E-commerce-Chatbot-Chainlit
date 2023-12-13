#import required libraries
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQAWithSourcesChain,RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
 SystemMessagePromptTemplate,
 HumanMessagePromptTemplate)
from langchain.llms import CTransformers
from langchain.llms import LlamaCpp
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from chainlit import run_sync
from cohere import Client
#
import chainlit as cl
import PyPDF2
from io import BytesIO
from getpass import getpass
#
import os
from configparser import ConfigParser
env_config = ConfigParser()


# Retrieve the cohere api key from the environmental variables
def read_config(parser: ConfigParser, location: str) -> None:
 assert parser.read(location), f"Could not read config {location}"
#
CONFIG_FILE = os.path.join(".", ".env")
read_config(env_config, CONFIG_FILE)
api_key = env_config.get("cohere", "api_key").strip()
os.environ["COHERE_API_KEY"] = api_key


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)

system_template = """You are and assistant fashion bot, your job is help the person to get there right outfit, and try to 
provide image get from the database for them, if you don't have that piece of cloths, just sorry the person and ask them for more
request, after you recommend them for a outfit, try to ask them if they want to try-on their choosen outfit using some button provided
 below.
 
Begin!
 - - - - - - - - 
{summaries}"""
messages = [SystemMessagePromptTemplate.from_template(system_template),HumanMessagePromptTemplate.from_template("{question}"),]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

@cl.action_callback("rating_button")
async def on_action(action):
    await action.remove()
    await cl.Message(content=f"Your outfit score based on my provided knowledge: 10").send()

@cl.action_callback("inter_virtual_fittingroom")
async def on_action(action):
    await action.remove()
    await cl.Message(content=f"Now you can try on your outfit! Have a nice experimence").send()

async def show_button():
    # Sending an action button within a chatbot message
    actions = [
        cl.Action(name="rating_button", value="example_value", description="Click me!"),
        cl.Action(name="enter_virtual_fittingroom", value="example_value", description="Click me!")
    ]

    await cl.Message(content="You can use these function for more experimence:", actions=actions).send()

#Decorator to react to the user websocket connection event.
@cl.on_chat_start
async def init():
    files = None
    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
        content="Please upload a PDF file to begin!",
        accept=["application/pdf"],
        ).send()
    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`…")
    await msg.send()
    # Read the PDF file
    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)
    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    # Create a Chroma vector store
    model_id = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceBgeEmbeddings(model_name= model_id,
    model_kwargs = {"device":"cpu"})
    #
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k=5
    # Store the embeddings in the user session
    cl.user_session.set("embeddings", embeddings)
    docsearch = await cl.make_async(Qdrant.from_texts)(
    texts, embeddings,location=":memory:", metadatas=metadatas
    )
    llm = LlamaCpp(streaming=True,
    model_path="zephyr-7b-beta.Q4_K_M.gguf",
    max_tokens = 1500,
    temperature=0.75,
    top_p=1,
    gpu_layers=0,
    stream=True,
    verbose=True,n_threads = int(os.cpu_count()/2),
    n_ctx=4096)
    #Hybrid Search
    qdrant_retriever = docsearch.as_retriever(search_kwargs={"k":5})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,qdrant_retriever],
    weights=[0.5,0.5])
    #Cohere Reranker
    #
    compressor = CohereRerank(client=Client(api_key=os.getenv("COHERE_API_KEY")),user_agent='langchain')
    #
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
    base_retriever=ensemble_retriever,
    )
    # Create a chain that uses the Chroma vector store
    chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    )
    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)
    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()
    #store the chain as long as the user session is active
    cl.user_session.set("chain", chain)

@cl.on_message
async def process_response(res:cl.Message):
    # retrieve the retrieval chain initialized for the current session
    chain = cl.user_session.get("chain") 
    # Chinlit callback handler
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    print("in retrieval QA")
    #res.content to extract the content from chainlit.message.Message
    print(f"res : {res.content}")
    
    special_word = ['try','look likes','fitting','wonder','score',"try",
    "look like",
    "test",
    "try out",
    "sample",
    "attempt",
    "experiment",
    "try on",
    "virtual fitting",
    "virtual room",
    "test outfit",
    "try outfit",
    "virtual try-on",
    "fitting room",
    "virtual dressing",
    "try clothes",
    "dress up",
    "try",
    "test",
    "check",
    "attempt",
    "experience",
    "experiment",
    "see how",
    "look like",
    "try out",
    "sample",
    "give a shot",
    "have a go",
    "have a look",
    "have a try",
    "test out",
    "explore",
    "examine",
    "assess",
    "evaluate",
    "try on",
    "virtual fitting",
    "virtual room",
    "test outfit",
    "try outfit",
    "virtual try-on",
    "fitting room",
    "virtual dressing",
    "try clothes",
    "dress up"]
    response = await chain.acall(res.content, callbacks=[cb])
    print(f"response: {response}")
    answer = response["result"]
    sources = response["source_documents"]
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources:
            print(source.metadata)
            try :
                source_name = source.metadata["source"]
            except :
                source_name = ""
            # Get the index of the source
            text = source.page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()   
        split_sentence = res.content.split()
        for word in split_sentence:
            if word in special_word:
                run_sync(show_button())