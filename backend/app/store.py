"""Chats and messages."""

from __future__ import annotations

import json

from . import db

TITLE_LENGTH = 60


def derive_title(question: str) -> str:
    text = " ".join(question.split())
    if len(text) <= TITLE_LENGTH:
        return text or "New chat"
    return text[:TITLE_LENGTH].rsplit(" ", 1)[0] + "…"


def list_chats() -> list[dict]:
    return db.query(
        """
        SELECT c.id, c.title, c.created_at, c.updated_at,
               (SELECT count(*) FROM messages m WHERE m.chat_id = c.id) AS message_count
        FROM chats c
        ORDER BY c.updated_at DESC
        """
    )


def create_chat(title: str = "New chat") -> dict:
    with db.connection() as conn:
        row = conn.execute(
            "INSERT INTO chats (title) VALUES (%s) RETURNING id, title, created_at, updated_at",
            (title,),
        ).fetchone()
    return row


def get_chat(chat_id: int) -> dict | None:
    return db.query_one("SELECT * FROM chats WHERE id = %s", (chat_id,))


def rename_chat(chat_id: int, title: str) -> None:
    db.execute(
        "UPDATE chats SET title = %s, updated_at = now() WHERE id = %s", (title, chat_id)
    )


def delete_chat(chat_id: int) -> None:
    db.execute("DELETE FROM chats WHERE id = %s", (chat_id,))


def list_messages(chat_id: int) -> list[dict]:
    return db.query(
        """
        SELECT id, role, content, citations, created_at
        FROM messages WHERE chat_id = %s ORDER BY created_at, id
        """,
        (chat_id,),
    )


def history_for_model(chat_id: int, limit: int = 6) -> list[dict]:
    """Recent turns, oldest first, in the shape chat models expect."""
    rows = db.query(
        """
        SELECT role, content FROM messages
        WHERE chat_id = %s ORDER BY created_at DESC, id DESC LIMIT %s
        """,
        (chat_id, limit),
    )
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def add_message(chat_id: int, role: str, content: str, citations: list | None = None) -> dict:
    with db.connection() as conn:
        row = conn.execute(
            """
            INSERT INTO messages (chat_id, role, content, citations)
            VALUES (%s, %s, %s, %s)
            RETURNING id, role, content, citations, created_at
            """,
            (chat_id, role, content, json.dumps(citations or [])),
        ).fetchone()
        conn.execute("UPDATE chats SET updated_at = now() WHERE id = %s", (chat_id,))
    return row
