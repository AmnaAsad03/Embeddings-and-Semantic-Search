from langchain_groq import ChatGroq

from src.generation.config import (
    GROQ_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    MIN_HYBRID_SCORE,
    TOP_K,
)

from src.generation.prompts import (
    MEDICAL_RAG_SYSTEM_PROMPT,
    MEDICAL_RAG_USER_PROMPT,
)


class MedicalAnswerGenerator:

    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

    def has_sufficient_evidence(self, results):
        if not results:
            return False

        valid_results = [
            result
            for result in results
            if result.get("hybrid_score", 0.0)
            >= MIN_HYBRID_SCORE
        ]

        return len(valid_results) > 0

    def format_evidence(self, results):
        evidence_blocks = []

        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})

            title = metadata.get(
                "title",
                "Unknown document"
            )

            source = metadata.get(
                "source",
                "Unknown source"
            )

            source_url = metadata.get(
                "source_url",
                ""
            )

            content = result.get(
                "content",
                ""
            )

            evidence_blocks.append(
                f"""
[Evidence {index}]
Document ID: {metadata.get("document_id", "Unknown")}
Title: {title}
Source: {source}
Source URL: {source_url}

Content:
{content}
"""
            )

        return "\n".join(evidence_blocks)

    def generate(self, question, results):
        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not self.has_sufficient_evidence(results):
            return {
                "answer": (
                    "I’m sorry, but the available medical sources "
                    "do not contain enough evidence to answer this "
                    "question reliably."
                ),
                "citations": [],
                "evidence_sufficient": False,
            }

        selected_results = results[:TOP_K]

        evidence = self.format_evidence(
            selected_results
        )

        user_prompt = MEDICAL_RAG_USER_PROMPT.format(
            question=question,
            evidence=evidence,
        )

        response = self.llm.invoke(
            [
                (
                    "system",
                    MEDICAL_RAG_SYSTEM_PROMPT,
                ),
                (
                    "human",
                    user_prompt,
                ),
            ]
        )

        return {
            "answer": response.content,
            "citations": self.extract_citations(
                selected_results
            ),
            "evidence_sufficient": True,
        }

    def extract_citations(self, results):
        """
        Extract unique document citations from retrieved results.

        Multiple chunks can come from the same document.
        Only one citation is returned for each unique document.
        """

        citations = []
        seen_documents = set()

        for result in results:
            metadata = result.get(
                "metadata",
                {}
            )

            document_id = metadata.get(
                "document_id"
            )

            title = metadata.get(
                "title"
            )

            source = metadata.get(
                "source"
            )

            source_url = metadata.get(
                "source_url"
            )

            unique_key = (
                document_id
                or source_url
                or f"{title}|{source}"
            )

            if unique_key in seen_documents:
                continue

            seen_documents.add(
                unique_key
            )

            citations.append(
                {
                    "citation_id": len(citations) + 1,
                    "document_id": document_id,
                    "title": title,
                    "source": source,
                    "source_url": source_url,
                }
            )

        return citations