import torch

from purellm.torchgpt.model import TinyGPT


@torch.no_grad()
def generate(
    model: TinyGPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
) -> list[int]:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # model.eval()

    # generated = torch.tensor(
    #     [prompt_ids],
    #     dtype=torch.long,
    #     device=device,
    # )

    # for _ in range(max_new_tokens):
    #     context = generated[:, -model.ctx_length :]
    #     with torch.autocast(
    #         device_type=context.device.type,
    #         dtype=autocast_dtype,
    #         enabled=autocast_dtype is not None,
    #     ):
    #         logits = model(context, use_cache=True)
    #     next_token_logits = logits[:, -1].float() / temperature
    #     probabilities = torch.softmax(next_token_logits, dim=-1)
    #     next_token = torch.multinomial(
    #         probabilities,
    #         num_samples=1,
    #     )
    #     context = next_token

    #     generated = torch.cat(
    #         [generated, next_token],
    #         dim=1,
    #     )

    model.eval()

    generated = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    model.reset_cache()
    context = generated[:, -model.ctx_length :]
    cached_length = 0

    try:
        for _ in range(max_new_tokens):
            with torch.autocast(
                device_type=context.device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                logits = model(context, use_cache=True)

            cached_length += context.shape[1]

            next_token_logits = logits[:, -1].float() / temperature
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            if cached_length == model.ctx_length:
                # ponytail: refill every step after reaching the context limit.
                model.reset_cache()
                context = generated[:, -model.ctx_length :]
                cached_length = 0
            else:
                context = next_token
    finally:
        model.reset_cache()

    return generated[0].tolist()
