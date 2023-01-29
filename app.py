import time
import nltk
import numpy as np
from cranfiled_data import CranDocuments, CranQueries, CranQueryReleventDocs
from boolen_retrieval_helper import TextPreProcessor
from vector_space_model_demo import DocStats
from markupsafe import escape

from flask import Flask, request, render_template

app = Flask(__name__)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


def get_documents_in_a_dict(cran_docs: CranDocuments):
    """Collect all abstracts (documents we are indexing)
    in a dictionary and return them.

    Args:
        cran_docs (CranDocuments): List containing instance
            of cranfield_data.Document object.

    Returns:
        dict: Containing doc.id as keys and document object as values
    """
    docs = {}
    for doc in cran_docs:
        docs[doc.id] = doc.abstract

    return docs


def get_document_objects_in_a_dict(cran_docs: CranDocuments):
    """Collect all documents we are indexing
    in a dictionary and return them.

    Args:
        cran_docs (CranDocuments): List containing instance
            of cranfield_data.Document object.

    Returns:
        dict: Containing doc.id as keys and document object as values
    """
    docs = {}
    for doc in cran_docs:
        docs[doc.id] = doc

    return docs


cran_docs = CranDocuments().get_all_docs()
cran_doc_objects = get_document_objects_in_a_dict(cran_docs)
cran_queries = CranQueries().get_all_queries()
cran_qrels = CranQueryReleventDocs().get_query_relevantdocs_map()
# print(cran_qrels)
all_docs = get_documents_in_a_dict(cran_docs)
# print(len(docs))
doc_stats = DocStats(all_docs)
# doc_stats = DocStatsScikit(docs)
print("# of documents: ", doc_stats.get_number_of_docs())
print("# of terms in the vocabulary: ", len(doc_stats.vocabulary))


@app.route("/")
def default():
    return render_template("search_input.html", host=request.hostname)


@app.route("/search", methods=["POST", "GET"])
def search():
    query = request.args.get("query", "")
    query = escape(query)
    pre_processed_query = TextPreProcessor(query).get_tokens()
    query = " ".join(pre_processed_query)
    q = doc_stats.get_unit_vector_for_text(query)
    doc_score_map = {}
    for doc_id in doc_stats.document_vector_map:
        score = np.dot(q, doc_stats.document_vector_map[doc_id])
        doc_score_map[doc_id] = score
    # Display top-k results
    k = 10
    sorted_docs = dict(
        sorted(doc_score_map.items(), key=lambda item: item[1], reverse=True)
    )
    doc_ids = []
    for doc_id in sorted_docs:
        doc_ids.append(doc_id)
        k -= 1
        if k == 0:
            break
    abstracts = [all_docs[doc_id] for doc_id in doc_ids]
    titles = [cran_doc_objects[doc_id].title for doc_id in doc_ids]
    return render_template("search_results.html", results=zip(titles, abstracts))
