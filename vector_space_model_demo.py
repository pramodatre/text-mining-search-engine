import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from boolen_retrieval_helper import TextPreProcessor


class DocStats:
    def __init__(self, docs: dict) -> None:
        self.docs = docs
        self.vocabulary = set()
        self.postings = {}
        self.tfs = {}
        self.idfs = {}
        # 1. tf and idf computation for each term in each document
        self.document_terms_map = self.document_terms_lookup(docs)
        self.document_tfidf_map = self.compute_tfidf()
        # 2. define a document vector which has one slot
        # for each term in the vocabulary
        self.doc_vector = pd.Series(
            data=list(range(len(self.vocabulary))), index=list(self.vocabulary)
        )
        # 3. tfidf scoring and populate the document vector
        self.document_vector_map = self.get_unit_vector_for_all_documents()

    def get_number_of_docs(self):
        return len(list(self.docs.keys()))

    def get_document_ids(self):
        return list(self.docs.keys())

    def get_unit_vector_for_all_documents(self):
        document_vector_map = {}
        for doc_id in self.get_document_ids():
            document_vector_map[doc_id] = self.get_document_unit_vector(doc_id)
        return document_vector_map

    def get_document_vector(self, doc_id):
        vector = np.zeros(len(self.vocabulary))
        term_tfidf_scores_map = self.document_tfidf_map[doc_id]
        for term in term_tfidf_scores_map:
            if term in self.doc_vector:
                vector[self.doc_vector[term]] = float(term_tfidf_scores_map[term])
        return vector

    def get_document_unit_vector(self, doc_id):
        vector = self.get_document_vector(doc_id)
        return vector / np.sqrt(np.sum(vector ** 2))

    def get_unit_vector_for_text(self, text):
        vector = np.zeros(len(self.vocabulary))
        tfs = self.term_frequency(text)
        for term in tfs:
            if term not in self.doc_vector:
                return vector
            idf = self.inverse_document_frequency(term)
            vector[self.doc_vector[term]] = tfs[term] * idf
        return vector / np.sqrt(np.sum(vector ** 2))

    def compute_tfidf(self):
        self.document_tfidf_map = {}
        for doc_id in self.document_terms_map:
            tfs = self.term_frequency(self.docs[doc_id])
            idfs = self.compute_idf(doc_id)
            tfidf_scores = {}
            for term in tfs:
                tfidf_scores[term] = tfs[term] * idfs[term]
            self.document_tfidf_map[doc_id] = tfidf_scores

        return self.document_tfidf_map

    def term_frequency(self, text: int) -> dict:
        # st.write(f"document: {document}")
        terms = TextPreProcessor(text).get_tokens()
        # st.write(f"pre-processed document: {terms}")
        return dict(Counter(terms))

    def document_terms_lookup(self, docs: dict) -> dict:
        document_terms_map = {}
        for doc_id in docs:
            terms = TextPreProcessor(docs[doc_id]).get_tokens()
            document_terms_map[doc_id] = set(terms)
            for term in terms:
                self.vocabulary.add(term)

        return document_terms_map

    def inverse_document_frequency(self, term: str) -> float:
        """
        This method implements inverse document frequence smooth
        which is given by log (N / 1 + n_{t}) + 1
        In practice, we may have terms in the query that may not
        occur in any document and we will have a zero in the
        denominator if we use the IDF definition log (N / n_{t})
        To avoid this, we are adding one to the denominator and
        adding one for the log value as shown above.

        Args:
            term (str): term for which we need to compute IDF

        Returns:
            float: IDF value for the supplied term
        """
        docs_containing_term = []
        total_docs = len(list(self.document_terms_map.keys()))
        for doc_id in self.document_terms_map:
            if term in self.document_terms_map[doc_id]:
                docs_containing_term.append(doc_id)

        return np.log10((total_docs + 1) / (len(docs_containing_term) + 1))

    def compute_idf(self, doc_id: str) -> dict:
        terms = self.document_terms_map[doc_id]
        for term in terms:
            self.idfs[term] = self.inverse_document_frequency(term)

        return self.idfs


def preprocess_text(text):
    return TextPreProcessor(text).get_tokens()


def launch_search_engine(doc_stats: DocStats):
    while True:
        query = input("Enter your search query: ")
        if query == "":
            break
        q = doc_stats.get_unit_vector_for_text(query)
        for doc_id in doc_stats.document_vector_map:
            score = np.dot(q, doc_stats.document_vector_map[doc_id])
            print(f"Document:{doc_id} has score: {score}")


if __name__ == "__main__":
    docs = {
        0: "experimental investigation of the aerodynamics of a wing in a slipstream .",
        1: "propeller slipstream effects as determined from wing pressure distribution on a large-scale six-propeller vtol model at static thrust .",
        2: "slipstream flow around several tilt-wing vtol aircraft models operating near the ground .",
        3: "investigation of the effects of ground proximity and propeller position on the effectiveness of a wing with large chord slotted flaps in redirecting propeller slipstream downward for vertical take-off .",
        4: "investigation of effectiveness of large chord slotted flaps in deflecting propeller slipstreams downward for vertical take-off and low-speed flight .",
    }
    doc_stats = DocStats(docs)
    doc_ids = doc_stats.get_document_ids()
    for doc_id in doc_ids:
        print(f"Document {doc_id}: {docs[doc_id]}")

    # Find dot product of document vectors
    checked_pairs = set()
    N = doc_stats.get_number_of_docs()
    similarity_matrix = np.ones((N, N))
    for doc1_id in doc_stats.document_vector_map:
        for doc2_id in doc_stats.document_vector_map:
            pair_present_in_checked_pairs = (doc2_id, doc1_id) in checked_pairs or (
                doc1_id,
                doc2_id,
            ) in checked_pairs
            if doc1_id == doc2_id or pair_present_in_checked_pairs:
                continue
            else:
                dot_product = np.dot(
                    doc_stats.document_vector_map[doc1_id],
                    doc_stats.document_vector_map[doc2_id],
                )
                print(f"{doc1_id}, {doc2_id}: {dot_product}")
                checked_pairs.add((doc1_id, doc2_id))
                similarity_matrix[doc1_id, doc2_id] = dot_product
                similarity_matrix[doc2_id, doc1_id] = dot_product

    sns.heatmap(similarity_matrix, annot=True, linewidth=0.5, fmt=".2f")
    plt.show()
    # launch_search_engine(doc_stats)
