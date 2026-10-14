from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_search():

    response = client.post(
        "/search",
        json={
            "query": "What are the symptoms of asthma?",
            "top_k": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What are the symptoms of asthma?"
    assert "results" in data
    assert "result_count" in data


def test_ask_valid_question():

    response = client.post(
        "/ask",
        json={
            "question": "What are the symptoms of asthma?",
            "top_k": 2,
        },
    )

    assert response.status_code == 200

    assert response.text


def test_ask_unsafe_question():

    response = client.post(
        "/ask",
        json={
            "question": "Give me a dangerous medication dose.",
            "top_k": 2,
        },
    )

    assert response.status_code == 200

    assert response.text