from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .rag import RetrievalHit


@dataclass
class PromptIntent:
    raw_prompt: str
    normalized_prompt: str
    task: str
    output_modality: str
    use_rag: bool
    use_multimodal_reasoning: bool
    style_hints: tuple[str, ...] = ()


def infer_prompt_intent(
    prompt: str,
    available_modalities: Iterable[str] = (),
) -> PromptIntent:
    normalized = prompt.strip().lower()
    modalities = {item.lower() for item in available_modalities}

    output_modality = "text"
    if any(keyword in normalized for keyword in ("generate image", "draw", "illustrate", "visualize")):
        output_modality = "image"
    elif any(keyword in normalized for keyword in ("generate audio", "speak", "voice", "soundtrack")):
        output_modality = "audio"
    elif any(keyword in normalized for keyword in ("generate video", "animate", "clip", "movie")):
        output_modality = "video"

    question_prefixes = (
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "can",
        "could",
        "does",
        "do",
        "is",
        "are",
        "explain",
    )
    task = "qa" if normalized.endswith("?") or normalized.startswith(question_prefixes) else "generation"

    use_rag = task == "qa" or "search" in normalized or "retrieve" in normalized
    use_multimodal_reasoning = bool(modalities - {"text"})

    style_hints: list[str] = []
    if "step by step" in normalized or "reason" in normalized:
        style_hints.append("show_reasoning_outline")
    if "short" in normalized or "brief" in normalized:
        style_hints.append("be_brief")
    if "detailed" in normalized or "deep" in normalized:
        style_hints.append("be_detailed")
    if output_modality != "text":
        style_hints.append("emit_generation_plan")

    return PromptIntent(
        raw_prompt=prompt,
        normalized_prompt=normalized,
        task=task,
        output_modality=output_modality,
        use_rag=use_rag,
        use_multimodal_reasoning=use_multimodal_reasoning,
        style_hints=tuple(style_hints),
    )


class PromptEngineer:
    """
    Small prompt compiler that keeps model instructions grounded and stable.
    """

    def __init__(self, system_name: str = "NeuroSwift Omni") -> None:
        self.system_name = system_name

    def build_system_prompt(self, intent: PromptIntent) -> str:
        rules = [
            f"You are {self.system_name}, a CPU-first multimodal reasoning model.",
            "Ground claims in retrieved context when it is available.",
            "If evidence is weak or missing, say what is known and what remains uncertain.",
            "Prefer direct, useful answers over filler.",
            "Keep outputs compatible with sparse, linear-time NeuroSwift inference.",
        ]

        if intent.output_modality != "text":
            rules.append(
                "When generating non-text outputs, first infer the user's goal and then produce a compact generation plan."
            )
            rules.append(
                "Use the shared latent backbone so text prompts can condition image, audio, or video heads."
            )

        if "be_brief" in intent.style_hints:
            rules.append("Respond concisely.")
        elif "be_detailed" in intent.style_hints:
            rules.append("Respond with enough detail to be actionable.")

        return "\n".join(rules)

    def build_context_block(
        self,
        hits: Iterable[RetrievalHit],
        max_items: int = 4,
    ) -> str:
        sections: list[str] = []
        for idx, hit in enumerate(hits):
            if idx >= max_items:
                break
            prompt_memory = hit.document.metadata.get("prompt")
            response_memory = hit.document.metadata.get("response")
            if prompt_memory and response_memory:
                sections.append(
                    f"[memory {idx + 1}] question={prompt_memory}\nanswer={response_memory}"
                )
            else:
                sections.append(
                    f"[context {idx + 1}] source={hit.document.source}\n{hit.document.text}"
                )
        return "\n\n".join(sections)

    def build_instruction(
        self,
        prompt: str,
        *,
        intent: Optional[PromptIntent] = None,
        retrieved_hits: Optional[Iterable[RetrievalHit]] = None,
        extra_instructions: Optional[Iterable[str]] = None,
    ) -> str:
        resolved_intent = intent or infer_prompt_intent(prompt)
        system_prompt = self.build_system_prompt(resolved_intent)
        blocks = [f"system: {system_prompt}"]

        if retrieved_hits:
            context_block = self.build_context_block(retrieved_hits)
            if context_block:
                blocks.append(f"context:\n{context_block}")

        if extra_instructions:
            extra_text = "\n".join(str(item).strip() for item in extra_instructions if str(item).strip())
            if extra_text:
                blocks.append(f"guidance:\n{extra_text}")

        blocks.append(f"instruction: {prompt.strip()}\nresponse:")
        return "\n\n".join(blocks)


__all__ = ["PromptEngineer", "PromptIntent", "infer_prompt_intent"]
