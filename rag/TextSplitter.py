from langchain.text_splitter import RecursiveCharacterTextSplitter

def Splitter(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for  p in pages)
    texts = text_splitter.split_text(context)
    return texts



