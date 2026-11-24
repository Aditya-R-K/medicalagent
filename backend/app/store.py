"""Chats, messages, conversation memory and token accounting."""

from __future__ import annotations

import json

from . import db

TITLE_LENGTH = 60
# Messages sent verbatim to the model; everything older lives in the chat's
# running summary. Keeps a long chat as cheap per question as a short one.
RECENT_MESSAGES = 4
# Assistant answers are long; the model needs their gist, not every word.
RECENT_ANSWER_CHARS = 1500
# Older messages are folded into the summary in batches, not one at a time.
FOLD_BATCH = 4


def derive_title(question: str) -> str:
    text = " ".join(question.split())
    if len(text) <= TITLE_LENGTH:
        return text or "New chat"
    return text[:TITLE_LENGTH].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------- #
# chats and messages
# --------------------------------------------------------------------------- #


def list_chats() -> list[dict]:
    return db.query(
        """
        SELECT c.id, c.title, c.created_at, c.updated_at,
               (SELECT count(*) FROM messages m WHERE m.chat_id = c.id) AS message_count,
               coalesce((SELECT sum(u.input_tokens + u.output_tokens)
                         FROM usage_events u WHERE u.chat_id = c.id), 0) AS tokens
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
        SELECT id, role, content, citations, usage, created_at
        FROM messages WHERE chat_id = %s ORDER BY created_at, id
        """,
        (chat_id,),
    )


def add_message(chat_id: int, role: str, content: str, citations: list | None = None,
                usage: dict | None = None) -> dict:
    with db.connection() as conn:
        row = conn.execute(
            """
            INSERT INTO messages (chat_id, role, content, citations, usage)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, role, content, citations, usage, created_at
            """,
            (chat_id, role, content, json.dumps(citations or []), json.dumps(usage or {})),
        ).fetchone()
        conn.execute("UPDATE chats SET updated_at = now() WHERE id = %s", (chat_id,))
    return row


# --------------------------------------------------------------------------- #
# conversation memory
# --------------------------------------------------------------------------- #


def conversation(chat_id: int) -> tuple[str, list[dict]]:
    """(running summary, recent messages) - what the model sees of the past.

    The last few messages go verbatim, answers trimmed to their opening;
    everything before them is represented by the summary alone, so the cost of
    a question no longer grows with the length of the chat.
    """
    chat = get_chat(chat_id) or {}
    rows = db.query(
        """
        SELECT role, content FROM messages
        WHERE chat_id = %s ORDER BY created_at DESC, id DESC LIMIT %s
        """,
        (chat_id, RECENT_MESSAGES),
    )
    recent = []
    for row in reversed(rows):
        content = row["content"]
        if row["role"] == "assistant" and len(content) > RECENT_ANSWER_CHARS:
            content = content[:RECENT_ANSWER_CHARS].rsplit(" ", 1)[0] + " …"
        recent.append({"role": row["role"], "content": content})
    return chat.get("summary") or "", recent


def history_for_model(chat_id: int, limit: int = RECENT_MESSAGES) -> list[dict]:
    """Recent turns, oldest first (kept for callers that want no summary)."""
    return conversation(chat_id)[1][-limit:]


def messages_to_fold(chat_id: int) -> list[dict]:
    """Messages older than the recent window that the summary does not cover
    yet - returned only once there are enough to be worth a summarising call."""
    chat = get_chat(chat_id)
    if not chat:
        return []
    rows = db.query(
        """
        SELECT id, role, content FROM messages
        WHERE chat_id = %s AND id > %s
        ORDER BY created_at, id
        """,
        (chat_id, chat.get("summarized_upto") or 0),
    )
    older = rows[:-RECENT_MESSAGES] if len(rows) > RECENT_MESSAGES else []
    return older if len(older) >= FOLD_BATCH else []


def save_summary(chat_id: int, summary: str, upto_message_id: int) -> None:
    db.execute(
        "UPDATE chats SET summary = %s, summarized_upto = %s WHERE id = %s",
        (summary, upto_message_id, chat_id),
    )


# --------------------------------------------------------------------------- #
# token accounting
# --------------------------------------------------------------------------- #


def record_usage(records: list[dict], chat_id: int | None = None,
                 message_id: int | None = None, document_id: int | None = None) -> None:
    if not records:
        return
    with db.connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO usage_events
                    (chat_id, message_id, document_id, step, provider, model,
                     input_tokens, output_tokens, cached_tokens, cache_write_tokens, calls)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (chat_id, message_id, document_id, r.get("step") or "call",
                     r.get("provider") or "", r.get("model") or "",
                     int(r.get("input") or 0), int(r.get("output") or 0),
                     int(r.get("cached") or 0), int(r.get("cache_write") or 0),
                     int(r.get("calls") or 1))
                    for r in records
                ],
            )


_TOTALS = """
    coalesce(sum(input_tokens), 0)       AS input,
    coalesce(sum(output_tokens), 0)      AS output,
    coalesce(sum(cached_tokens), 0)      AS cached,
    coalesce(sum(cache_write_tokens), 0) AS cache_write,
    coalesce(sum(calls), 0)              AS calls
"""


def chat_usage(chat_id: int) -> dict:
    return db.query_one(f"SELECT {_TOTALS} FROM usage_events WHERE chat_id = %s", (chat_id,))


def usage_report(days: int = 14) -> dict:
    """Everything the analytics panel shows, in one round of queries."""
    def totals(where: str = "TRUE") -> dict:
        return db.query_one(f"SELECT {_TOTALS} FROM usage_events WHERE {where}")

    return {
        "all_time": totals(),
        "today": totals("created_at >= date_trunc('day', now())"),
        "week": totals("created_at >= now() - interval '7 days'"),
        "chats": totals("step NOT IN ('index')"),
        "indexing": totals("step = 'index'"),
        "daily": db.query(
            f"""
            SELECT to_char(d.day, 'YYYY-MM-DD') AS day,
                   coalesce(sum(u.input_tokens), 0)  AS input,
                   coalesce(sum(u.output_tokens), 0) AS output,
                   coalesce(sum(u.cached_tokens), 0) AS cached
            FROM generate_series(date_trunc('day', now()) - interval '{int(days) - 1} days',
                                 date_trunc('day', now()), interval '1 day') AS d(day)
            LEFT JOIN usage_events u ON date_trunc('day', u.created_at) = d.day
            GROUP BY d.day ORDER BY d.day
            """
        ),
        "by_model": db.query(
            f"""
            SELECT provider, model, {_TOTALS}
            FROM usage_events GROUP BY provider, model
            ORDER BY sum(input_tokens + output_tokens) DESC LIMIT 12
            """
        ),
        "by_step": db.query(
            f"""
            SELECT step, {_TOTALS}
            FROM usage_events GROUP BY step
            ORDER BY sum(input_tokens + output_tokens) DESC
            """
        ),
        "top_chats": db.query(
            f"""
            SELECT u.chat_id, coalesce(c.title, '(deleted chat)') AS title, {_TOTALS},
                   count(DISTINCT u.message_id) AS answers
            FROM usage_events u LEFT JOIN chats c ON c.id = u.chat_id
            WHERE u.step <> 'index'
            GROUP BY u.chat_id, c.title
            ORDER BY sum(u.input_tokens + u.output_tokens) DESC LIMIT 8
            """
        ),
    }
