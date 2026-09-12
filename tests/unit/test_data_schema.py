import json

from clinhallu.data.preprocessing import load_and_validate_json


def test_medhallu_knowledge_list_is_normalized(tmp_path):
    source = tmp_path / "eval.json"
    source.write_text(
        json.dumps(
            [
                {
                    "Question": "What is supported?",
                    "Knowledge": ["First fact.", "Second fact."],
                    "Answer": "A supported answer.",
                    "hallu_label": 0,
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_and_validate_json(str(source))

    assert len(records) == 1
    assert records[0]["question"] == "What is supported?"
    assert records[0]["context"] == "First fact.\n\nSecond fact."
    assert records[0]["hall_label"] == 0
