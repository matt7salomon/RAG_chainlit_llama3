import os
import io
import sys
import logging
import numpy as np
import chainlit as cl
from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv, dotenv_values
from langchain_community.embeddings import AzureOpenAIEmbeddings, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_groq import ChatGroq
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
np.float_ = np.float64
class ChatbotApp:
    def __init__(self):
        self.load_environment_variables()
        self.configure_logger()
        self.configure_system_prompt()
        self.files = None
        self.all_texts = []

    def load_environment_variables(self):
        if os.path.exists(".env"):
            load_dotenv(override=True)
            self.config = dotenv_values(".env")

        self.temperature = float(os.environ.get("TEMPERATURE", 0.9))
        self.api_base = os.getenv("OPENAI_API_KEY")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_type = os.environ.get("AZURE_OPENAI_TYPE", "none")
        self.api_version = os.environ.get("AZURE_OPENAI_VERSION", "2023-12-01-preview")
        self.chat_completion_deployment = 'llama3-8b-8192'
        self.embeddings_deployment = 'llama3'
        self.model = os.getenv("AZURE_OPENAI_MODEL")
        self.max_size_mb = int(os.getenv("CHAINLIT_MAX_SIZE_MB", 100))
        self.max_files = int(os.getenv("CHAINLIT_MAX_FILES", 10))
        self.text_splitter_chunk_size = int(os.getenv("TEXT_SPLITTER_CHUNK_SIZE", 1000))
        self.text_splitter_chunk_overlap = int(os.getenv("TEXT_SPLITTER_CHUNK_OVERLAP", 10))
        self.embeddings_chunk_size = int(os.getenv("EMBEDDINGS_CHUNK_SIZE", 16))
        self.max_retries = int(os.getenv("MAX_RETRIES", 5))
        self.retry_min_seconds = int(os.getenv("RETRY_MIN_SECONDS", 1))
        self.retry_max_seconds = int(os.getenv("RETRY_MAX_SECONDS", 5))
        self.timeout = int(os.getenv("TIMEOUT", 30))
        self.debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
        self.groq_api_key = os.getenv("groq_api_key")
        self.OLLAMA_URL = 'http://127.0.0.1:11434'

    def configure_logger(self):
        logging.basicConfig(
            stream=sys.stdout,
            format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

    def configure_system_prompt(self):
        system_template = """Use the following pieces of context to answer the users question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        ALWAYS return a "SOURCES" part in your answer.
        The "SOURCES" part should be a reference to the source of the document from which you got your answer.

        Example of your response should be:

        ```
        The answer is foo
        SOURCES: xyz
        ```

        Begin!
        ----------------
        {summaries}"""
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        self.prompt = ChatPromptTemplate.from_messages(messages)
        self.chain_type_kwargs = {"prompt": self.prompt}

    async def start_chat(self):
        await self.wait_for_files()
        await self.process_files()
        await self.create_vector_store()
        await self.create_chain()
        await self.notify_user_files_processed()

    async def wait_for_files(self):
        while self.files is None:
            self.files = await cl.AskFileMessage(
                content=f"Please upload up to {self.max_files} `.pdf` or `.docx` files to begin.",
                accept=[
                    "application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ],
                max_size_mb=self.max_size_mb,
                max_files=self.max_files,
                timeout=86400,
                raise_on_timeout=False,
            ).send()

    async def process_files(self):
        content = f"Processing {', '.join([f'`{f.name}`' for f in self.files])}..."
        self.logger.info(content)
        msg = cl.Message(content=content, author="Chatbot")
        await msg.send()

        for file in self.files:
            with open(file.path, "rb") as uploaded_file:
                file_contents = uploaded_file.read()

            self.logger.info("[%d] bytes were read from %s", len(file_contents), file.path)
            bytes_io = io.BytesIO(file_contents)
            extension = file.name.split(".")[-1]
            text = self.read_file(bytes_io, extension)
            self.split_text_into_chunks(text)

    def read_file(self, bytes_io, extension):
        text = ""
        if extension == "pdf":
            reader = PdfReader(bytes_io)
            for i in range(len(reader.pages)):
                text += reader.pages[i].extract_text()
                if self.debug:
                    self.logger.info("[%s] read from %s", text, file.path)
        elif extension == "docx":
            doc = Document(bytes_io)
            paragraph_list = [paragraph.text for paragraph in doc.paragraphs]
            text = "\n".join(paragraph_list)
            if self.debug:
                self.logger.info("[%s] read from %s", text, file.path)
        return text

    def split_text_into_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.text_splitter_chunk_size,
            chunk_overlap=self.text_splitter_chunk_overlap,
        )
        texts = text_splitter.split_text(text)
        self.all_texts.extend(texts)

    async def create_vector_store(self):
        metadatas = [{"source": f"{i}-pl"} for i in range(len(self.all_texts))]
        embeddings = self.get_embeddings()
        self.db = await cl.make_async(Chroma.from_texts)(
            self.all_texts, embeddings, metadatas=metadatas
        )
        cl.user_session.set("metadatas", metadatas)
        cl.user_session.set("texts", self.all_texts)

    def get_embeddings(self):
        if self.api_type == "azure":
            return AzureOpenAIEmbeddings(
                openai_api_version=self.api_version,
                openai_api_type=self.api_type,
                openai_api_key=self.api_key,
                azure_endpoint=self.api_base,
                azure_deployment=self.embeddings_deployment,
                max_retries=self.max_retries,
                retry_min_seconds=self.retry_min_seconds,
                retry_max_seconds=self.retry_max_seconds,
                chunk_size=self.embeddings_chunk_size,
                timeout=self.timeout,
            )
        else:
            return OllamaEmbeddings(
                model=self.embeddings_deployment,
            )

    async def create_chain(self):
        if self.api_type == "azure":
            llm = ChatOpenAI(
                openai_api_type=self.api_type,
                openai_api_version=self.api_version,
                openai_api_key=self.api_key,
                azure_endpoint=self.api_base,
                temperature=self.temperature,
                azure_deployment=self.chat_completion_deployment,
                streaming=True,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
        else:
            llm = ChatGroq(model_name=self.chat_completion_deployment, groq_api_key=self.groq_api_key)

        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs=self.chain_type_kwargs,
        )
        cl.user_session.set("chain", self.chain)

    async def notify_user_files_processed(self):
        content = f"{', '.join([f'`{f.name}`' for f in self.files])} processed. You can now ask questions."
        self.logger.info(content)
        msg = cl.Message(content=content, author="Chatbot")
        await msg.update()

    async def handle_message(self, message):
        chain = cl.user_session.get("chain")
        cb = cl.AsyncLangchainCallbackHandler()
        response = await chain.acall(message.content, callbacks=[cb])
        self.logger.info("Question: [%s]", message.content)

        answer = response["answer"]
        sources = response["sources"].strip()
        source_elements = []

        if self.debug:
            self.logger.info("Answer: [%s]", answer)

        metadatas = cl.user_session.get("metadatas")
        all_sources = [m["source"] for m in metadatas]
        texts = cl.user_session.get("texts")

        if sources:
            found_sources = []
            for source in sources.split(","):
                source_name = source.strip().replace(".", "")
                try:
                    index = all_sources.index(source_name)
                except ValueError:
                    continue
                text = texts[index]
                found_sources.append(source_name)
                source_elements.append(cl.Text(content=text, name=source_name))

            if found_sources:
                answer += f"\nSources: {', '.join(found_sources)}"
            else:
                answer += "\nNo sources found"

        await cl.Message(content=answer, elements=source_elements).send()


app = ChatbotApp()

@cl.on_chat_start
async def start():
    await app.start_chat()

@cl.on_message
async def main(message: cl.Message):
    await app.handle_message(message)
