from surrealdb import RecordID, Value


def extract_id(record: Value) -> str:
    if isinstance(record, RecordID):
        return str(record.id)  # pyright: ignore[reportAny]
    elif isinstance(record, dict) and "id" in record:
        rec_id = record["id"]
        id = rec_id.id if isinstance(rec_id, RecordID) else rec_id  # pyright: ignore[reportAny]
        return str(id)
    else:
        raise ValueError(
            "Invalid record, expected a RecordID or a dict with an 'id' key"
        )
