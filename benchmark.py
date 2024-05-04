import argparse
from typing import List

import ollama
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from datetime import datetime


class Message(BaseModel):
    role: str
    content: str


class OllamaResponse(BaseModel):
    model: str
    created_at: datetime
    message: Message
    done: bool
    total_duration: int
    load_duration: int = 0
    prompt_eval_count: int = Field(-1, validate_default=True)
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

    @field_validator("prompt_eval_count")
    @classmethod
    def validate_prompt_eval_count(cls, value: int) -> int:
        if value == -1:
            print(
                "\nWarning: prompt token count was not provided, potentially due to prompt caching. For more info, see https://github.com/ollama/ollama/issues/2068\n"
            )
            return 0  # Set default value
        return value


def run_benchmark(model_name: str, prompt: str, verbose: bool) -> OllamaResponse:

    last_element = None

    if verbose:
        stream = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
        )
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
            last_element = chunk
    else:
        last_element = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

    if not last_element:
        print("System Error: No response received from ollama")
        return None

    # with open("data/ollama/ollama_res.json", "w") as outfile:
    #     outfile.write(json.dumps(last_element, indent=4))

    return OllamaResponse.model_validate(last_element)


def nanosec_to_sec(nanosec):
    return nanosec / 1000000000


def inference_stats(model_response: OllamaResponse):
    # Use properties for calculations
    prompt_ts = model_response.prompt_eval_count / (
        nanosec_to_sec(model_response.prompt_eval_duration)
    )
    response_ts = model_response.eval_count / (
        nanosec_to_sec(model_response.eval_duration)
    )
    total_ts = (model_response.prompt_eval_count + model_response.eval_count) / (
        nanosec_to_sec(
            model_response.prompt_eval_duration + model_response.eval_duration
        )
    )

    print(
        f"""
----------------------------------------------------
        {model_response.model}
        \tPrompt eval: {prompt_ts:.2f} t/s
        \tResponse: {response_ts:.2f} t/s
        \tTotal: {total_ts:.2f} t/s

        Stats:
        \tPrompt tokens: {model_response.prompt_eval_count}
        \tResponse tokens: {model_response.eval_count}
        \tModel load time: {nanosec_to_sec(model_response.load_duration):.2f}s
        \tPrompt eval time: {nanosec_to_sec(model_response.prompt_eval_duration):.2f}s
        \tResponse time: {nanosec_to_sec(model_response.eval_duration):.2f}s
        \tTotal time: {nanosec_to_sec(model_response.total_duration):.2f}s
----------------------------------------------------
        """
    )


def average_stats(responses: List[OllamaResponse]):
    if len(responses) == 0:
        print("No stats to average")
        return

    res = OllamaResponse(
        model=responses[0].model,
        created_at=datetime.now(),
        message=Message(
            role="system",
            content=f"Average stats across {len(responses)} runs",
        ),
        done=True,
        total_duration=sum(r.total_duration for r in responses),
        load_duration=sum(r.load_duration for r in responses),
        prompt_eval_count=sum(r.prompt_eval_count for r in responses),
        prompt_eval_duration=sum(r.prompt_eval_duration for r in responses),
        eval_count=sum(r.eval_count for r in responses),
        eval_duration=sum(r.eval_duration for r in responses),
    )
    print("Average stats:")
    inference_stats(res)


def get_benchmark_models(skip_models: List[str] = []) -> List[str]:
    models = ollama.list().get("models", [])
    model_names = [model["name"] for model in models]
    if len(skip_models) > 0:
        model_names = [model for model in model_names if model not in skip_models]
    print(f"Evaluating models: {model_names}\n")
    return model_names


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks on your Ollama models."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--skip-models",
        nargs="*",
        default=[],
        help="List of model names to skip. Separate multiple models with spaces.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="*",
        default=[
           "In a population of giraffes, an environmental change occurs that favors individuals that are tallest. As a result, more of the taller individuals are able to obtain nutrients and survive to pass along their genetic information. This is an example of\nA. directional selection.\nB. stabilizing selection.\nC. sexual selection.\nD. disruptive selection.",
           "Which of the changes below following the start codon in an mRNA would most likely have the greatest deleterious effect?\nA. a deletion of a single nucleotide\nB. a deletion of a nucleotide triplet\nC. a single nucleotide substitution of the nucleotide occupying the first codon position\nD. a single nucleotide substitution of the nucleotide occupying the third codon position",
           "A feature of amino acids NOT found in carbohydrates is the presence of\nA. carbon atoms\nB. oxygen atoms\nC. nitrogen atoms\nD. hydrogen atoms",
           "Let A and B be two sets of words (strings) from \u03a3*, for some alphabet of symbols \u03a3. Suppose that B is a subset of A. Which of the following statements must always be true of A and B ?\nI. If A is finite, then B is finite.\nII. If A is regular, then B is regular.\nIII. If A is context-free, then B is context-free.\nA. I only\nB. II only\nC. III only\nD. I and II only",
           "The plates of a capacitor are charged to a potential difference of 5 V. If the capacitance is 2 mF, what is the charge on the positive plate?\nA. 0.005 C\nB. 0.01 C\nC. 0.02 C\nD. 0.5 C",
           "Traveling at an initial speed of 1.5 \u00d7 10^6 m\/s, a proton enters a region of constant magnetic field, B, of magnitude 1.0 T. If the proton's initial velocity vector makes an angle of 30\u00b0 with the direction of B, compute the proton's speed 4 s after entering the magnetic field.\nA. 5.0 \u00d7 10^5 m\/s\nB. 7.5 \u00d7 10^5 m\/s\nC. 1.5 \u00d7 10^6 m\/s\nD. 3.0 \u00d7 10^6 m\/s",
           "A group G in which (ab)^2 = a^2b^2 for all a, b in G is necessarily\nA. finite\nB. cyclic\nC. of order two\nD. abelian",
           "The set of all real numbers under the usual multiplication operation is not a group since\nA. multiplication is not a binary operation\nB. multiplication is not associative\nC. identity element does not exist\nD. zero has no inverse",
           "What is the difference between a direct leak and a side channel?\nA. A direct leak creates a denial of service by failing to free memory, while a channel frees memory as a side effect\nB. A direct leak is one that is intentional, rather than by unintentional\nC. A direct leak comes via the software system's intended interaction mechanism, where as a side channel leak comes from measurements of other system features, like timing, power usage, or space usage\nD. There is no difference",
           "The four Primary Security Principles related to messages are\nA. Confidentiality, Integrity, Non repudiation and Authentication\nB. Confidentiality, Access Control, Integrity, Non repudiation\nC. Authentication, Authorization, Availability, Integrity\nD. Availability, Authorization, Confidentiality, Integrity"
        ],
        help="List of prompts to use for benchmarking. Separate multiple prompts with spaces.",
    )

    args = parser.parse_args()

    verbose = args.verbose
    skip_models = args.skip_models
    prompts = args.prompts
    print(f"\nVerbose: {verbose}\nSkip models: {skip_models}\nPrompts: {prompts}")

    model_names = get_benchmark_models(skip_models)
    benchmarks = {}

    for model_name in model_names:
        responses: List[OllamaResponse] = []
        for prompt in prompts:
            if verbose:
                print(f"\n\nBenchmarking: {model_name}\nPrompt: {prompt}")
            response = run_benchmark(model_name, prompt, verbose=verbose)
            responses.append(response)

            if verbose:
                print(f"Response: {response.message.content}")
                inference_stats(response)
        benchmarks[model_name] = responses

    for model_name, responses in benchmarks.items():
        average_stats(responses)


if __name__ == "__main__":
    main()
    # Example usage:
    # python benchmark.py --verbose --skip-models aisherpa/mistral-7b-instruct-v02:Q5_K_M llama2:latest --prompts "What color is the sky" "Write a report on the financials of Microsoft"
