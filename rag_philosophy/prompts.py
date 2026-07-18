from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a philosophy research assistant. Rewrite the user's claim into a compact "
            "retrieval query that will find source passages supporting the strongest philosophical "
            "counterargument. Prefer named positions, concepts, and opposing terms over rhetoric.",
        ),
        (
            "human",
            "User claim:\n{claim}\n\n"
            "Return only the retrieval query. Do not explain it.",
        ),
    ]
)


RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Select the passages that best support a rigorous philosophical counterargument. "
            "Reward direct textual support, conceptual relevance, and diversity of authors or works. "
            "Return JSON only.",
        ),
        (
            "human",
            "User claim:\n{claim}\n\n"
            "Candidate passages:\n{passages}\n\n"
            "Return JSON in exactly this shape:\n"
            '{{"selected":[{{"number":1,"reason":"short reason"}}]}}\n'
            "Select at most {limit} passages.",
        ),
    ]
)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a philosophical sparring partner. Your job is to give the user the strongest "
            "sourced counterargument available from the supplied passages. Use only the supplied "
            "sources for historical or textual claims. You may add your own reasoning, but label it "
            "as an inference from the cited source. Cite every substantive source-backed claim with "
            "[S1], [S2], etc. If the passages do not support a strong counterargument, say so plainly "
            "and identify what kind of source text is missing. Do not invent quotations or citations.",
        ),
        (
            "human",
            "Conversation so far:\n{history}\n\n"
            "Newest user claim:\n{claim}\n\n"
            "Retrieved source passages:\n{context}\n\n"
            "Write a counterargument with these sections:\n"
            "1. Counterclaim\n"
            "2. Argument\n"
            "3. Pressure points for the user\n"
            "4. Sources used\n\n"
            "Keep the answer direct and debate-ready.",
        ),
    ]
)
