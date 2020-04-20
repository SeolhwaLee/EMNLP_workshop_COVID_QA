from fastapi import FastAPI
from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.elasticsearch import ElasticsearchRetriever

app = FastAPI()

data_sources = ["scraped", "internal"]
bert_models = [model_name.strip() for model_name in open("models.txt")] + ["BM25"]

finders = {}
for data_source in data_sources:
    for model_name in bert_models:
        if model_name != "BM25":
            docStore = ElasticsearchDocumentStore(
                host="localhost",
                username="",
                password="",
                index="%s_%s" % (model_name, data_source),
                search_fields="question",
                text_field="answer",
                doc_id_field="id",
                embedding_field="question_emb",
                embedding_dim=768 if "base" in model_name else 1024,
                excluded_meta_data=["question_emb"])
            retriever = ElasticsearchRetriever(document_store=docStore,
                                               embedding_model=model_name,
                                               model_format="sentence_transformers")
        else:
            retriever = ElasticsearchRetriever(document_store=docStore)

        finder = Finder(reader=None, retriever=retriever)
        finders[(data_source, model_name)] = finder


@app.get("/")
def root(question: str = "What is Covid-19?", data_source="scraped", model_name="bert-base-nli-mean-tokens"):
    if data_source not in data_sources:
        return {"error": "data_source is invalid."}
    elif model_name not in bert_models:
        return {"error": "model_name is invalid."}
    return finders[(data_source, model_name)].get_answers_via_similar_questions(question=question, top_k_retriever=5)


@app.get("/models")
async def get_models():
    return {"available_models": bert_models}


@app.get("/data_sources")
async def get_models():
    return {"data_sources": data_sources}
