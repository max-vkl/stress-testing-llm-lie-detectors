"""
Tokenizes text datasets.
"""

import re
from dataclasses import dataclass, replace
from typing import Any, Self, cast

import torch
from jaxtyping import Bool, Int
from termcolor import colored
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from deception_detection.data import Dialogue
from deception_detection.data.base import DialogueDataset
from deception_detection.gemma4_chat import is_gemma4_tokenizer, render_gemma4_dialogue
from deception_detection.log import logger
from deception_detection.models import ModelType
from deception_detection.types import Message
from deception_detection.utils import preprocess_dialogue

_EOT_TOKEN_BY_MODEL_TYPE: dict[ModelType, str] = {
    "gemma": "<end_of_turn>",
    "mistral": "</s>",
    "llama": "<|eot_id|>",
    "qwen": "<|im_end|>",
}


@dataclass(frozen=True)
class _ResolvedEOT:
    text: str
    token_id: int


def decode_single_token(token_id: int, tokenizer: PreTrainedTokenizerBase) -> str:
    """Decode a single token."""
    # Use the same tokenizer type detection logic as the rest of the codebase
    tokenizer_path_lower = tokenizer.name_or_path.lower()
    if "llama" in tokenizer_path_lower:
        # see https://github.com/TransformerLensOrg/TransformerLens/issues/373
        return tokenizer.decode([891, token_id])[3:]
    return tokenizer.decode([token_id])


def get_str_tokens(token_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> list[list[str]]:
    # Ensure the input is on CPU and convert to a list of lists
    token_ids_list = token_ids.cpu().tolist()

    # Convert token IDs to string tokens
    string_tokens = []
    for dialogue in token_ids_list:
        dialogue_tokens = [tokenizer.bos_token]
        for token_id in dialogue:
            # we use bos as pad
            if token_id != tokenizer.bos_token_id:
                token = decode_single_token(cast(int, token_id), tokenizer)
                dialogue_tokens.append(token)
        string_tokens.append(dialogue_tokens)
    return string_tokens


def get_dialogues_with_updated_system_prompts(formatted_dialogues: list[str], dialogues: list[Dialogue], system_prompt_start: str="[SYSTEM_PROMPT]", system_prompt_end: str="[/SYSTEM_PROMPT]"):
    for i in range(len(formatted_dialogues)):
        system_prompt = formatted_dialogues[i].split(system_prompt_start)[1].split(system_prompt_end)[0]
        for idx, msg in enumerate(dialogues[i]):
            if msg.role == "system":
                dialogues[i][idx] = replace(msg, content=system_prompt, detect=False)
        # Check if system message doesn't exist at all
        if not any(msg.role == "system" for msg in dialogues[i]):
            dialogues[i] = [Message("system", system_prompt, detect=False)] + dialogues[i]

    assert all(len([m for m in d if m.role == "system"]) == 1 for d in dialogues), f"All dialogues must have exactly one system message"

    return dialogues

@dataclass
class TokenizedDataset:
    """
    A class to represent tokenized input data for deception detection models. If the dataset
    contains a pair of truthful/deceptive dialogues, this class only represents ONE of those.

    Attributes:
        dialogues (list[Dialogue]): List of dialogues that have been tokenized.
        formatted_dialogues (list[str]): List of dialogues formatted by the tokenizer.
        tokens (Int[Tensor, "batch seq_pos"]): Tokenized input text.
        detection_mask (Bool[Tensor, "batch seq_pos"]): Mask over tokens for where we will compute
            detection scores.
        attention_mask (Bool[Tensor, "batch seq_pos"]): Attention mask for the tokenized text.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used to process the text.
    """

    dialogues: list[Dialogue]
    formatted_dialogues: list[str]
    tokens: Int[Tensor, "n_dialogues seq_pos"]
    str_tokens: list[list[str]]
    detection_mask: Bool[Tensor, "n_dialogues seq_pos"] | None
    attention_mask: Int[Tensor, "n_dialogues seq_pos"]
    dataset_partial_id: str | None = None

    @staticmethod
    def _final_message_is_assistant(dialogue: Dialogue) -> bool:
        """Check if the final message in a dialogue is an assistant message."""
        return dialogue[-1].role == "assistant"

    @staticmethod
    def _resolve_eot_token(
        tokenizer: PreTrainedTokenizerBase, tokenizer_type: ModelType
    ) -> _ResolvedEOT:
        """Resolve and validate the model's exact, single-token end-of-turn marker.

        New tokenizers can declare ``eot_token`` and ``eot_token_id`` (Gemma 4 does). Older
        tokenizers use the model-family fallback below (Gemma 3 does). In either case, encoding
        must produce exactly the declared token ID. Taking the first ID of a multi-token string is
        intentionally forbidden because that silently points the detector at an unrelated token.
        """
        tokenizer_name = str(getattr(tokenizer, "name_or_path", type(tokenizer).__name__))
        declared_text = getattr(tokenizer, "eot_token", None)
        declared_id = getattr(tokenizer, "eot_token_id", None)

        if (declared_text is None) != (declared_id is None):
            raise ValueError(
                f"Tokenizer {tokenizer_name!r} has incomplete EOT metadata: "
                f"eot_token={declared_text!r}, eot_token_id={declared_id!r}"
            )

        if declared_text is None:
            eot_text = _EOT_TOKEN_BY_MODEL_TYPE[tokenizer_type]
        elif not isinstance(declared_text, str) or not declared_text:
            raise ValueError(
                f"Tokenizer {tokenizer_name!r} has invalid eot_token={declared_text!r}"
            )
        else:
            eot_text = declared_text

        tokenizer_output = tokenizer(eot_text, add_special_tokens=False)
        encoded_ids = tokenizer_output["input_ids"]
        if isinstance(encoded_ids, Tensor):
            encoded_ids = encoded_ids.tolist()
        if not isinstance(encoded_ids, list | tuple) or any(
            isinstance(token_id, list | tuple) for token_id in encoded_ids
        ):
            raise ValueError(
                f"Tokenizer {tokenizer_name!r} returned an invalid encoding for EOT token "
                f"{eot_text!r}: {encoded_ids!r}"
            )
        try:
            encoded_ids = [int(token_id) for token_id in encoded_ids]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Tokenizer {tokenizer_name!r} returned non-integer IDs for EOT token "
                f"{eot_text!r}: {encoded_ids!r}"
            ) from exc

        if len(encoded_ids) != 1:
            raise ValueError(
                f"EOT token {eot_text!r} for tokenizer {tokenizer_name!r} must encode to "
                f"exactly one token, but encoded to {encoded_ids!r}"
            )

        encoded_id = encoded_ids[0]
        if declared_id is not None:
            try:
                expected_id = int(declared_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Tokenizer {tokenizer_name!r} has invalid eot_token_id={declared_id!r}"
                ) from exc
            if encoded_id != expected_id:
                raise ValueError(
                    f"Tokenizer {tokenizer_name!r} has inconsistent EOT metadata: "
                    f"eot_token {eot_text!r} encodes to {encoded_id}, not eot_token_id "
                    f"{expected_id}"
                )

        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if unk_token_id is not None and encoded_id == int(unk_token_id):
            raise ValueError(
                f"EOT token {eot_text!r} for tokenizer {tokenizer_name!r} resolves to the "
                "unknown-token ID"
            )

        return _ResolvedEOT(text=eot_text, token_id=encoded_id)

    @staticmethod
    def _terminal_eot_suffix(
        formatted_dialogue: str, eot: _ResolvedEOT, dialogue_idx: int
    ) -> str:
        """Return the exact terminal EOT suffix, or fail if the chat template omitted it."""
        for suffix in (f"{eot.text}\n", eot.text):
            if formatted_dialogue.endswith(suffix):
                return suffix
        raise ValueError(
            f"Dialogue {dialogue_idx} formatted by the tokenizer does not end with the "
            f"resolved EOT marker {eot.text!r}; trailing text is "
            f"{formatted_dialogue[-80:]!r}"
        )

    @classmethod
    def _strip_terminal_eot(
        cls, formatted_dialogue: str, eot: _ResolvedEOT, dialogue_idx: int
    ) -> str:
        """Remove a terminal EOT for generation, never silently leaving an unknown suffix."""
        suffix = cls._terminal_eot_suffix(formatted_dialogue, eot, dialogue_idx)
        return formatted_dialogue[: -len(suffix)]

    @staticmethod
    def _prefix_pattern(tokenizer_type: ModelType) -> re.Pattern[str]:
        """Regular expression that matches chat template tokens, used to match the detect mask
        to token indexes.

        Generally needs to match all of:
        - the start of dialogue, including padding, bos token, and the first header.
        - the <eot> token for end of the last message, plus the header for the new message
        - '\n\n' as the separator between a folded system prompt and the user message
        - an empty string for two messages objects from the same user that are merged.
        """
        if tokenizer_type == "mistral":
            """
            Example:
            '<s>[SYSTEM_PROMPT]You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\nYour knowledge base was last updated on 2023-10-01. The current date is 2025-06-25.\n\nWhen you\'re not sure about some information, you say that you don\'t have the information and don\'t make up anything.\nIf the user\'s question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?")[/SYSTEM_PROMPT][INST]Pretend you\'re an honest person making statements about the world.[/INST]The Titanic was a famous ship that sank in 1912.</s>'
            """

            return re.compile(
                r"<s>\[SYSTEM_PROMPT\]|\[/SYSTEM_PROMPT\]\[INST\]|\[/INST\]|</s>\[INST\]|^"
            )
        elif tokenizer_type == "llama":
            begin_of_text = r"((<\|pad\|>)*(<\|begin_of_text\|>))?"
            header = r"<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n"
            end_of_turn = r"<\|eot_id\|>"
            # llama3 system prompt has some boilerplate which we keep constant across all prompts
            date_info = (
               r"(Cutting Knowledge Date: December 2023\nToday Date: \d\d \w\w\w 202[45]\n\n)?"
            )
            # matches:
            # - padding + BOT + system prompt boilerplate
            # - end of turn + next header
            # - padding + BOT + system prompt boilerplate + end of turn + next header (if we have
            #   no additional system prompt)
            # - '\n\n' or the empty string
            return re.compile(
                rf"({begin_of_text}{header}{date_info})?({end_of_turn}{header})?|(\n\n)?"
            )
        elif tokenizer_type == "qwen":
            """
            Example:
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nPretend you're an honest person making statements about the world.<|im_end|>\n<|im_start|>assistant\nThe Titanic was a famous ship that sank in 1912.<|im_end|>\n"
            """
            # Qwen ChatML format matches:
            # 1. Start of dialogue: <|im_start|>role\n
            # 2. Between messages: <|im_end|>\n<|im_start|>role\n
            # 3. Folded prompt separator: \n\n
            # 4. End of dialogue: empty string

            return re.compile(
                r"<\|im_start\|>(system|user|assistant)\n"  # start of dialogue
                r"|<\|im_end\|>\n<\|im_start\|>(system|user|assistant)\n"  # between messages
                r"|\n\n"
                r"|^"
            )
        elif tokenizer_type == "gemma":
            """
            Gemma 2/3 fold system messages into the first user turn; Gemma 4 uses a native
            system turn and an empty thought-channel generation prefix. Folded prompts use
            '\n\n'. Gemma 3 and Gemma 4 use different
            turn markers, both handled here:
              Gemma 3: <start_of_turn>role\n ... <end_of_turn>\n
              Gemma 4: <|turn>role\n ... <turn|>\n
            Example (Gemma 3):
            "<bos><start_of_turn>user\nYou are a helpful AI assistant for travel tips and recommendations\n\nWhat can you help me with?<end_of_turn>\n<start_of_turn>model\nWith planning murder!<end_of_turn>\n"
            Example (Gemma 4):
            "<bos><|turn>system\nYou are a helpful assistant.<turn|>\n"
            "<|turn>user\nWhat can you help me with?<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>Sure!<turn|>\n"
            """
            sot = r"(<start_of_turn>|<\|turn>)"  # start-of-turn marker (Gemma 3 | Gemma 4)
            eot = r"(<end_of_turn>|<turn\|>)"  # end-of-turn marker (Gemma 3 | Gemma 4)
            header = rf"{sot}(system|developer|user|model)\n"
            thought = r"(?:<\|channel>thought\n<channel\|>)?"
            # Gemma chat format matches:
            # 1. Start of dialogue: (padding)(<bos>)<sot>role\n
            # 2. Between messages: <eot>\n<sot>role\n
            # 3. Folded system->user separator: \n\n
            # 4. Merged same-role messages / end of dialogue: empty string
            return re.compile(
                rf"(<pad>)*(<bos>)?{header}{thought}"  # start of dialogue
                rf"|{eot}\n{header}{thought}"  # between messages
                r"|\n\n"
                r"|^"
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    @staticmethod
    def _get_detection_mask(
        tokens: Int[Tensor, "num_dialogues seq"],
        tokenizer_out: Any,
        dialogues: list[Dialogue],
        formatted_dialogues: list[str],
        for_generation: bool,
        detect_all: bool,
        padding: dict[str, int],
        tokenizer_type: ModelType,
    ) -> Bool[Tensor, "num_dialogues seq"] | None:
        if for_generation:
            return None
        elif detect_all:
            return torch.ones_like(tokens).bool()

        detection_mask = torch.zeros_like(tokens).bool()
        for batch_idx, dialogue in enumerate(dialogues):
            char_idx: int = 0

            for message in dialogue:
                # find the start and end of the message in the formatted text
                match = re.match(TokenizedDataset._prefix_pattern(tokenizer_type), formatted_dialogues[batch_idx][char_idx:])
                assert match is not None, f"text={formatted_dialogues[batch_idx][char_idx:]}"
                start_char_idx: int = char_idx + match.end()
                content = message.content
                # Native templates may trim leading whitespace (Llama/Gemma 3), while
                # other renderers preserve it (Gemma 4 and adjacent assistant segments).
                # Align only the mask; never change the dialogue or rendered model input.
                if not formatted_dialogues[batch_idx].startswith(content, start_char_idx):
                    content = content.lstrip()
                end_char_idx: int = start_char_idx + len(content)

                assert (formatted_dialogues[batch_idx][start_char_idx:end_char_idx] == content), (formatted_dialogues[batch_idx][start_char_idx:end_char_idx], message.content)

                # use tokenizer utilities to find the corresponding tokens
                if message.detect:
                    start_tok_index = tokenizer_out.char_to_token(batch_idx, start_char_idx)
                    assert (
                        start_tok_index is not None
                    ), f"start_char_idx={start_char_idx} text={message.content}, {dialogue}"

                    # Get the initial end token index using standard logic
                    end_tok_index = tokenizer_out.char_to_token(batch_idx, end_char_idx)

                    # Special handling for Qwen tokenizers: manually extend by 1 token for last assistant message
                    if tokenizer_type == "qwen":
                        # Check if this is the last assistant message in the dialogue
                        is_last_assistant = (message.role == "assistant" and message == dialogue[-1])

                        if is_last_assistant and end_tok_index is not None:
                            # For Qwen's last assistant message, extend the detection mask by one more token
                            # to include the last token that's often missed due to char_to_token mapping issues
                            end_tok_index += 1

                    # we also optionally include the activations that predict the first tok of
                    # the message.
                    detection_mask[
                        batch_idx,
                        start_tok_index - padding["left"] : end_tok_index + padding["right"],
                    ] = True

                # update our current position in the string
                char_idx = end_char_idx

        return detection_mask

    @staticmethod
    def _detection_mask_at_eot(
        tokens: Int[Tensor, "num_dialogues seq"],
        detection_mask: Bool[Tensor, "num_dialogues seq"],
        attention_mask: Int[Tensor, "num_dialogues seq"],
        eot_token_id: int,
    ) -> Bool[Tensor, "num_dialogues seq"]:
        """Reduce each dialogue's detection mask to its relevant EOT token.

        LTLR uses one activation per dialogue. If an EOT is already detected, use the last
        detected EOT. Otherwise add the first EOT after the final detected token to the mask. A
        completely empty source mask falls back to the dialogue's terminal EOT; this covers data
        such as short REPE statements whose marked prefix tokenizes to no complete token.
        """
        eot_mask = (tokens == eot_token_id) & attention_mask.bool()
        output = torch.zeros_like(detection_mask)

        for row in range(tokens.shape[0]):
            eot_positions = torch.nonzero(eot_mask[row], as_tuple=False).flatten()
            if eot_positions.numel() == 0:
                raise ValueError(f"Dialogue {row} has no EOT token after tokenization")

            detected_eot_positions = torch.nonzero(
                eot_mask[row] & detection_mask[row], as_tuple=False
            ).flatten()
            if detected_eot_positions.numel() > 0:
                target = detected_eot_positions[-1]
            else:
                detected_positions = torch.nonzero(detection_mask[row], as_tuple=False).flatten()
                if detected_positions.numel() == 0:
                    target = eot_positions[-1]
                else:
                    later_eots = eot_positions[eot_positions > detected_positions[-1]]
                    if later_eots.numel() == 0:
                        raise ValueError(
                            f"Dialogue {row} has detected tokens but no following EOT token"
                        )
                    target = later_eots[0]
            output[row, target] = True

        selected_per_row = output.sum(dim=1)
        if not torch.all(selected_per_row == 1):
            raise RuntimeError(
                "Internal EOT-mask invariant failed: expected exactly one EOT per dialogue, got "
                f"{selected_per_row.cpu().tolist()}"
            )
        if not torch.all(tokens[output] == eot_token_id):
            raise RuntimeError("Internal EOT-mask invariant failed: selected token is not EOT")

        return output

    @staticmethod
    def _detection_mask_at_last_content(
        tokens: Int[Tensor, "num_dialogues seq"],
        detection_mask: Bool[Tensor, "num_dialogues seq"],
        attention_mask: Int[Tensor, "num_dialogues seq"],
        eot_token_id: int,
    ) -> Bool[Tensor, "num_dialogues seq"]:
        """Keep detected content tokens immediately followed by the validated EOT token.

        This is an explicitly requested compatibility mode. Every dialogue must yield at least
        one such token; otherwise execution stops instead of silently using another position.
        """
        next_is_eot = torch.zeros_like(detection_mask)
        if tokens.shape[1] > 1:
            next_is_eot[:, :-1] = (
                (tokens[:, 1:] == eot_token_id) & attention_mask[:, 1:].bool()
            )
        output = detection_mask & attention_mask.bool() & next_is_eot

        missing_rows = torch.nonzero(output.sum(dim=1) == 0, as_tuple=False).flatten()
        if missing_rows.numel() > 0:
            rows = missing_rows.cpu().tolist()
            raise ValueError(
                "Explicit last-content-token selection found no detected content immediately "
                f"before EOT token ID {eot_token_id} for dialogue rows {rows}; refusing to "
                "fall back to EOT or another content token"
            )

        return output

    @classmethod
    def from_dialogue_list(
        cls,
        dialogues: list[Dialogue],
        tokenizer: PreTrainedTokenizerBase,
        padding: dict[str, dict[str, int]],
        for_generation: bool = False,
        detect_all: bool = False,
        detect_only_start_of_turn: bool = False,
        detect_only_last_token: bool = False,
        detect_only_eot: bool = False,
        max_length: int | None = None,
        dataset_partial_id: str | None = None,
        trim_reasoning: bool = False,
        split_on: str | None = None,
        ensure_system_prompt: bool = False,
        preserve_trailing_whitespace: bool = False,
    ) -> Self:
        """
        Create an TokenizedDataset of a list of dialogues

        Args:
            dialogues (list[Dialogue]): List of dialogues to tokenize.
            tokenizer (PreTrainedTokenizerBase): Tokenizer to use for processing the text.
            padding (dict[str, dict[str, int]]): Padding configuration for detection mask. Non-zero
                values will extend the detection mask to include tokens just before or after the
                marked ones.
            for_generation (bool): Whether to have the tokens ready for generation.
                Adds the generation prompt if no assistant prompt present, else trims off the
                assistant end of turn.
            detect_all (bool): Whether to detect on all tokens rather than just marked ones.
            detect_only_start_of_turn (bool): Whether to have the detect-mask only be true at
                <|start_of_turn|> tokens.
            detect_only_last_token (bool): Whether to only detect at the last token of each message,
                the position just before <|eot_id|> tokens.
            detect_only_eot (bool): Whether to reduce each dialogue's mask to an EOT token. Used
                by LTLR, which reads one activation per dialogue.
            max_length (int | None): Maximum sequence length for truncation. None means no
                truncation.
            preserve_trailing_whitespace (bool): Preserve message-final whitespace that a chat
                template would otherwise trim.
        """
        assert dialogues, "Cannot tokenize an empty dialogue"
        if detect_only_eot and detect_only_last_token:
            raise ValueError("detect_only_eot and detect_only_last_token are mutually exclusive")
        if for_generation and (detect_only_eot or detect_only_last_token):
            raise ValueError(
                "EOT/last-content detection masks cannot be requested in generation mode"
            )

        # Extract tokenizer type from path (handles both original and merged model paths)
        tokenizer_path_lower = tokenizer.name_or_path.lower()
        if "mistral" in tokenizer_path_lower:
            tokenizer_type = "mistral"
        elif "llama" in tokenizer_path_lower:
            tokenizer_type = "llama"
        elif "qwen" in tokenizer_path_lower:
            tokenizer_type = "qwen"
        elif "gemma" in tokenizer_path_lower:
            tokenizer_type = "gemma"
        else:
            raise NotImplementedError(f"Only gemma, mistral, llama, and qwen tokenizers are supported {tokenizer.name_or_path}")

        # if llama and no system prompt, throw a warning
        if (
            ensure_system_prompt
            and tokenizer_type == "llama"
            and not any(d[0].role == "system" for d in dialogues)
        ):
            logger.warning(
                f"No system prompts for {dataset_partial_id}. "
                f"Adding a default. e.g >>{dialogues[0]}<<"
            )
            dialogues = [
                [Message("system", "You are an AI assistant.", False)] + d for d in dialogues
            ]

        # Gemma 2/3 fold the system into the first user turn; Gemma 4 has a native system role.
        native_gemma4 = tokenizer_type == "gemma" and is_gemma4_tokenizer(tokenizer)
        fold_system = tokenizer_type == "gemma" and not native_gemma4
        processed = [preprocess_dialogue(d, fold_system=fold_system) for d in dialogues]

        # Several model chat templates apply Jinja's trim filter to message content. Canonical RepE
        # includes prefixes whose final tokenizer token represents whitespace, so temporarily put
        # a non-whitespace sentinel after those suffixes and remove it after template rendering.
        whitespace_sentinel = "__DD_PRESERVE_TRAILING_WHITESPACE_7E9F3A__"
        if preserve_trailing_whitespace:
            for dialogue in processed:
                for message in dialogue:
                    content = message["content"]
                    if whitespace_sentinel in content:
                        raise ValueError("Trailing-whitespace sentinel collides with message text")
                    if content and content[-1].isspace():
                        message["content"] = content + whitespace_sentinel

        final_assistants = [cls._final_message_is_assistant(d) for d in dialogues]
        assert all(final_assistants[0] == fa for fa in final_assistants)
        final_assistant = final_assistants[0]
        needs_eot = (
            (for_generation and final_assistant) or detect_only_eot or detect_only_last_token
        )
        eot = cls._resolve_eot_token(tokenizer, tokenizer_type) if needs_eot else None

        if native_gemma4:
            formatted_dialogues = [
                render_gemma4_dialogue(
                    tokenizer, dialogue,
                    add_generation_prompt=for_generation and not final_assistant,
                )
                for dialogue in processed
            ]
            if for_generation and final_assistant:
                assert eot is not None
                formatted_dialogues = [
                    cls._strip_terminal_eot(dialogue, eot, dialogue_idx)
                    for dialogue_idx, dialogue in enumerate(formatted_dialogues)
                ]
        elif for_generation:
            formatted_dialogues: list[str] = tokenizer.apply_chat_template(
                processed, tokenize=False, add_generation_prompt=not final_assistant
            )  # type: ignore
            if final_assistant:
                assert eot is not None
                formatted_dialogues = [
                    cls._strip_terminal_eot(dialogue, eot, dialogue_idx)
                    for dialogue_idx, dialogue in enumerate(formatted_dialogues)
                ]
        else:
            formatted_dialogues: list[str] = tokenizer.apply_chat_template(
                processed, tokenize=False, add_generation_prompt=False
            )  # type: ignore

        if preserve_trailing_whitespace:
            formatted_dialogues = [
                dialogue.replace(whitespace_sentinel, "") for dialogue in formatted_dialogues
            ]

        if detect_only_eot or detect_only_last_token:
            assert eot is not None
            for dialogue_idx, formatted_dialogue in enumerate(formatted_dialogues):
                cls._terminal_eot_suffix(formatted_dialogue, eot, dialogue_idx)

        # For the following models tokenizer.apply_chat_template adds a system prompt when there is none. We want to add them therefore also to the original dialogues
        if tokenizer_type == "qwen":
            dialogues = get_dialogues_with_updated_system_prompts(formatted_dialogues, dialogues, "<|im_start|>system\n", "<|im_end|>")
        elif tokenizer_type == "mistral":
            dialogues = get_dialogues_with_updated_system_prompts(formatted_dialogues, dialogues, "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]")

        tokenizer_out = tokenizer(
            formatted_dialogues,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )

        tokens: torch.Tensor = tokenizer_out["input_ids"]  # type: ignore
        detection_mask = cls._get_detection_mask(
            tokens,
            tokenizer_out,
            dialogues,
            formatted_dialogues,
            for_generation,
            detect_all,
            padding[tokenizer_type],
            tokenizer_type,
        )
        if detection_mask is not None and detect_only_start_of_turn:
            marker = "<|turn>" if native_gemma4 else "<start_of_turn>"
            marker_ids = tokenizer(marker, add_special_tokens=False).input_ids
            if native_gemma4 and len(marker_ids) != 1:
                raise ValueError("Gemma 4 start-of-turn marker must encode to one token")
            sot_tok = cast(int, marker_ids[0])
            if native_gemma4:
                turn_mask = torch.zeros_like(detection_mask)
                for row in range(len(tokens)):
                    starts = torch.nonzero(
                        detection_mask[row] & ~torch.cat([
                            torch.zeros(1, dtype=torch.bool, device=tokens.device),
                            detection_mask[row, :-1],
                        ]), as_tuple=False,
                    ).flatten()
                    for start in starts.tolist():
                        previous = torch.nonzero(tokens[row, :start + 1] == sot_tok).flatten()
                        if previous.numel():
                            turn_mask[row, previous[-1]] = True
                detection_mask = turn_mask
            else:
                detection_mask = detection_mask & (tokens == sot_tok)
        attention_mask = cast(Tensor, tokenizer_out["attention_mask"])
        if detection_mask is not None and detect_only_eot:
            assert eot is not None
            detection_mask = cls._detection_mask_at_eot(
                tokens,
                detection_mask,
                attention_mask,
                eot.token_id,
            )
        if detection_mask is not None and detect_only_last_token:
            assert eot is not None
            detection_mask = cls._detection_mask_at_last_content(
                tokens,
                detection_mask,
                attention_mask,
                eot.token_id,
            )

        if detection_mask is not None and detection_mask.sum() == 0:
            logger.warning("No tokens in detection mask")

        return cls(
            dialogues=dialogues,
            formatted_dialogues=formatted_dialogues,
            tokens=tokens,
            str_tokens=get_str_tokens(tokens, tokenizer),
            detection_mask=detection_mask,
            attention_mask=cast(Tensor, tokenizer_out["attention_mask"]),
            dataset_partial_id=dataset_partial_id,
        )

    @classmethod
    def from_bare_statements(
        cls,
        dialogues: list[Dialogue],
        tokenizer: PreTrainedTokenizerBase,
        *,
        detect_only_eot: bool,
        detect_only_last_token: bool,
        max_length: int | None = None,
        dataset_partial_id: str | None = None,
    ) -> Self:
        """Encode supplied statements directly, with one content or appended-EOT readout.

        Native special-token handling matches ``tokenizer.encode(statement)``. No chat
        renderer, role header, instruction, or thinking wrapper enters these model inputs.
        Padding is added only after choosing the readout, and cannot become a fit row.
        """
        if not dialogues or detect_only_eot == detect_only_last_token:
            raise ValueError("Bare statements require data and exactly one content/EOT readout")
        name = tokenizer.name_or_path.lower()
        model_type = next(
            (family for family in ("llama", "gemma", "mistral", "qwen") if family in name), None
        )
        if model_type is None:
            raise ValueError(f"Unsupported bare-statement tokenizer: {tokenizer.name_or_path}")
        eot = (
            cls._resolve_eot_token(tokenizer, cast(ModelType, model_type))
            if detect_only_eot
            else None
        )
        rows: list[list[int]] = []
        for dialogue in dialogues:
            if len(dialogue) != 1 or dialogue[0].role != "assistant":
                raise ValueError("Bare datasets must contain exactly one assistant statement")
            statement = dialogue[0].content
            if not statement.strip():
                raise ValueError("Cannot read an empty bare statement")
            ids = tokenizer.encode(statement, add_special_tokens=True)
            if not ids or ids[-1] in tokenizer.all_special_ids:
                raise ValueError("Native bare encoding must end in a statement-content token")
            if eot is not None:
                ids = [*ids, eot.token_id]
            if max_length is not None and len(ids) > max_length:
                raise ValueError(
                    "Bare statement exceeds max_length; refusing to truncate its readout"
                )
            rows.append(ids)

        pad_id = tokenizer.pad_token_id
        if not isinstance(pad_id, int):
            raise ValueError("Bare-statement batches require a padding token")
        width = max(map(len, rows))
        tokens = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(tokens)
        detection_mask = torch.zeros_like(tokens, dtype=torch.bool)
        for index, ids in enumerate(rows):
            start = width - len(ids) if tokenizer.padding_side == "left" else 0
            tokens[index, start : start + len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[index, start : start + len(ids)] = 1
            detection_mask[index, start + len(ids) - 1] = True
        return cls(
            dialogues=dialogues,
            formatted_dialogues=[
                tokenizer.decode(ids, clean_up_tokenization_spaces=False) for ids in rows
            ],
            tokens=tokens,
            str_tokens=[[decode_single_token(token, tokenizer) for token in ids] for ids in rows],
            detection_mask=detection_mask,
            attention_mask=attention_mask,
            dataset_partial_id=dataset_partial_id,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: DialogueDataset,
        tokenizer: PreTrainedTokenizerBase,
        for_generation: bool = False,
        detect_all: bool = False,
        detect_only_start_of_turn: bool = False,
        detect_only_last_token: bool = False,
        detect_only_eot: bool = False,
        max_length: int | None = None,
        additional_id: str | None = None,
    ) -> Self:
        """
        Create a TokenizedDataset from a DialogueDataset

        Args:
            dataset (DialogueDataset): The dataset containing dialogues to tokenize.
            tokenizer (PreTrainedTokenizerBase): Tokenizer to use for processing the text.
            padding (dict[str, dict[str, int]]): Padding configuration.
            for_generation (bool): Whether to prepare tokens for generation.
            detect_all (bool): Whether to detect all tokens.
            detect_only_start_of_turn (bool): Whether to detect only the start of turn tokens.
            detect_only_last_token (bool): Whether to detect only the last token before EOT.
            detect_only_eot (bool): Whether to detect only the EOT associated with the final
                detected content.
            max_length (int | None): Maximum length of tokenized sequences.

        Returns:
            Self: A new instance of TokenizedDataset.
        """
        # A tokenizer-constructed dataset may specify its readout position. Keep this
        # dataset-local so canonical training/validation does not change unrelated eval masks.
        detect_only_last_token = detect_only_last_token or bool(
            getattr(dataset, "detect_only_last_token", False)
        )
        preserve_trailing_whitespace = bool(
            getattr(dataset, "preserve_trailing_whitespace", False)
        )
        dataset_partial_id = f"{dataset.base_name}__{dataset.variant}"
        if additional_id:
            dataset_partial_id += f"_{additional_id}"
        if getattr(dataset, "bare_statements", False):
            if for_generation or detect_all or detect_only_start_of_turn:
                raise ValueError("Bare TIU datasets support only content/EOT probe readouts")
            return cls.from_bare_statements(
                dataset.dialogues,
                tokenizer,
                detect_only_eot=detect_only_eot,
                detect_only_last_token=detect_only_last_token,
                max_length=max_length,
                dataset_partial_id=dataset_partial_id,
            )
        return cls.from_dialogue_list(
            dialogues=dataset.dialogues,
            tokenizer=tokenizer,
            padding=dataset.padding,
            for_generation=for_generation,
            detect_all=detect_all,
            detect_only_start_of_turn=detect_only_start_of_turn,
            detect_only_last_token=detect_only_last_token,
            detect_only_eot=detect_only_eot,
            max_length=max_length,
            dataset_partial_id=dataset_partial_id,
            preserve_trailing_whitespace=preserve_trailing_whitespace,
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int | slice) -> Self:
        idx = idx if isinstance(idx, slice) else slice(idx, idx + 1)
        return self.__class__(
            dialogues=self.dialogues[idx],
            formatted_dialogues=self.formatted_dialogues[idx],
            tokens=self.tokens[idx],
            str_tokens=self.str_tokens[idx],
            detection_mask=self.detection_mask[idx] if self.detection_mask is not None else None,
            attention_mask=self.attention_mask[idx],
        )

    def display_detection_mask(self, idx: int, tokenizer: PreTrainedTokenizerBase) -> None:
        """Display the detection mask for a given dialogue."""
        assert self.detection_mask is not None, "Detection mask not available"
        readable_tokens: list[str] = tokenizer.batch_decode(
            self.tokens[idx], clean_up_tokenization_spaces=False
        )
        for readable_token, mask in zip(readable_tokens, self.detection_mask[idx], strict=True):
            readable_token = readable_token.replace("\n", "\\n")
            color = "red" if mask else "light_grey"
            print(colored(readable_token, color), end="")
            print("|", end="")


def get_tokenized_dataset_pair(
    honest: DialogueDataset,
    deceptive: DialogueDataset,
    tokenizer: PreTrainedTokenizerBase,
    enforce_equal_detect_lengths: bool = False,
    detect_all: bool = False,
    detect_only_start_of_turn: bool = False,
    detect_only_last_token: bool = False,
    detect_only_eot: bool = False,
) -> tuple[TokenizedDataset, TokenizedDataset]:
    """
    Tokenizes the input dialogues in a PairedDialogueDataset, returning a pair of TokenizedDatasets.

    Args:
        honest (DialogueDataset): Dataset containing truthful dialogues.
        deceptive (DialogueDataset): Dataset containing deceptive dialogues.
        tokenizer (PreTrainedTokenizerBase): Tokenizer to use for processing the text.
        enforce_equal_detect_lengths (bool): If True, this will trim the detection mask to the
            shorter of the pair.
        detect_all (bool): Whether to detect all tokens.
        detect_only_start_of_turn (bool): Whether to detect only the start of turn tokens.
        detect_only_last_token (bool): Whether to detect only the last token before EOT.
        detect_only_eot (bool): Whether to detect only the EOT associated with the final detected
            content.

    Returns:
        tuple[TokenizedDataset, TokenizedDataset]: A pair of TokenizedDataset instances.
    """

    truthful_tokenized = TokenizedDataset.from_dataset(
        honest,
        tokenizer,
        detect_all=detect_all,
        detect_only_start_of_turn=detect_only_start_of_turn,
        detect_only_last_token=detect_only_last_token,
        detect_only_eot=detect_only_eot,
        additional_id="honest",
    )
    deceptive_tokenized = TokenizedDataset.from_dataset(
        deceptive,
        tokenizer,
        detect_all=detect_all,
        detect_only_start_of_turn=detect_only_start_of_turn,
        detect_only_last_token=detect_only_last_token,
        detect_only_eot=detect_only_eot,
        additional_id="deceptive",
    )

    if enforce_equal_detect_lengths:
        equalize_detection_masks(truthful_tokenized, deceptive_tokenized)

    return truthful_tokenized, deceptive_tokenized


def equalize_detection_masks(dataset1: TokenizedDataset, dataset2: TokenizedDataset):
    """
    Modifies the detection masks of both datasets in place to ensure they have equal lengths.
    """
    assert dataset1.detection_mask is not None
    assert dataset2.detection_mask is not None
    for i in range(len(dataset1.detection_mask)):
        mask1 = dataset1.detection_mask[i]
        mask2 = dataset2.detection_mask[i]

        start1, end1 = get_mask_range(mask1)
        start2, end2 = get_mask_range(mask2)

        length1 = end1 - start1
        length2 = end2 - start2
        min_length = min(length1, length2)

        # Trim the detection masks to the minimum length
        dataset1.detection_mask[i][start1 + min_length : end1] = False
        dataset2.detection_mask[i][start2 + min_length : end2] = False


def get_mask_range(mask: torch.Tensor) -> tuple[int, int]:
    """
    Finds the start and end indices of the True values in the mask.
    """
    start = torch.argmax((mask == True).int())
    end = start + torch.argmax((mask[start:] == False).int())

    # If no end found, set end to the length of the mask
    if end == start:
        end = len(mask)

    start = int(start)
    end = int(end)

    return start, end
