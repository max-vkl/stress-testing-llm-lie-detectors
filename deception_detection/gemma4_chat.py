"""Gemma 4 rendering shared by generation, probing, and probe cache identity."""

from __future__ import annotations

from typing import Any

GEMMA4_CHAT_RENDERING_VERSION = 1


def is_gemma4_tokenizer(tokenizer: Any) -> bool:
    # The native marker also identifies local/renamed tokenizer directories.
    return (
        getattr(tokenizer, "eot_token", None) == "<turn|>"
        or "gemma-4" in str(getattr(tokenizer, "name_or_path", "")).lower()
    )


def render_gemma4_dialogue(
    tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool = False
) -> str:
    """Teacher-force the final answer after exactly the prefix used to generate it.

    Gemma 4's ordinary full-conversation template omits the empty thought channel
    emitted by add_generation_prompt=True. Earlier turns should still use the
    native history representation; only the final assistant completion is continued.
    """
    native = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if messages[-1]["role"] != "assistant":
        return native
    eot = getattr(tokenizer, "eot_token", "<turn|>")
    suffix = next((s for s in (eot + "\n", eot) if native.endswith(s)), None)
    if suffix is None:
        raise ValueError("Gemma 4 chat template omitted its terminal end-of-turn marker")
    prefix = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prefix + messages[-1]["content"] + suffix
