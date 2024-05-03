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
            "A completely submerged object always displaces its own weight of fluid.",
            "When a diver points a flashlight upward toward the surface of the water at an angle 20° from the normal, the beam of light totally internally reflects",
            "According to four-dimensional geometry, the angles of a triangle add to 180°.",
            "A voltage will be induced in a wire loop when the magnetic field within that loop changes",
            "Among these colors, the one that has the most energy per photon is red",
            "Based on the characteristic population curves that result from plotting population growth of a species, the most effective means of controlling the mosquito population is to	maintain the population at a point corresponding to the midpoint of its logistic curve",
            "A frameshift mutation is created when	telomeric sequences are removed from DNA",
            "To prevent desiccation and injury, the embryos of terrestrial vertebrates are encased within a fluid secreted by the amnion",
            "Which of the following is a second messenger that stimulates release of calcium ions into the cytoplasm?	Prostaglandins",
            "Which of the following is a second messenger that stimulates release of calcium ions into the cytoplasm?	Prostaglandins",
            "The rate, r, of a zero-order chemical reaction A → B can be expressed as which of the following?"
            "Infrared (IR) spectroscopy is useful for determining certain aspects of the structure of organic molecules because	all molecular bonds absorb IR radiation",
            "Infrared (IR) spectroscopy is useful for determining certain aspects of the structure of organic molecules because	all molecular bonds absorb IR radiation",
            "When the following equation is balanced, which of the following is true?__ MnO4− + __ I− + __ H+ <-> __ Mn2+ + __ IO3− + __ H2O",
            "The equation ΔH = ΔU + PΔV is always applicable",
            "The access matrix approach to protection has the difficulty that the matrix, if stored directly, is large and can be clumsy to manage",
            "An integer c is a common divisor of two integers x and y if and only if c is a divisor of x and c is a divisor of y. Which of the following sets of integers could possibly be the set of all common divisors of two integers?	{-6,-2, -1, 1, 2, 6}",
            "You want to cluster 7 points into 3 clusters using the k-Means Clustering algorithm. Suppose after the first iteration, clusters C1, C2 and C3 contain the following two-dimensional points: C1 contains the 2 points: {(0,6), (6,0)} C2 contains the 3 points: {(2,2), (4,4), (6,6)} C3 contains the 2 points: {(5,5), (7,7)} What are the cluster centers computed for these 3 clusters?	C1: (3,3), C2: (4,4), C3: (6,6)",
            "Any set of Boolean operators that is sufficient to represent all Boolean expressions is said to be complete. Which of the following is NOT complete?"
            "Consider the collection of all undirected graphs with 10 nodes and 6 edges. Let M and m, respectively, be the maximum and minimum number of connected components in any graph in the collection. If a graph has no selfloops and there is at most one edge between any pair of nodes, which of the following is true?",
            "Let k be the number of real solutions of the equation e^x + x - 2 = 0 in the interval [0, 1], and let n be the number of real solutions that are not in [0, 1]. Which of the following is true?",
            "Up to isomorphism, how many additive abelian groups G of order 16 have the property that x + x + x + x = 0 for each x in G ?",
            "Suppose P is the set of polynomials with coefficients in Z_5 and degree less than or equal to 7. If the operator D sends p(x) in P to its derivative p′(x), what are the dimensions of the null space n and range r of D?",
            "The shortest distance from the curve xy = 8 to the origin is 4",
            "There are 25 suitcases, 5 of which are damaged. Three suitcases are selected at random. What is the probability that exactly 2 are damaged?",
            "The quantum efficiency of a photon detector is 0.1. If 100 photons are sent into the detector, one after the other, the detector will detect photons an average of 10 times, with an rms deviation of about 4",
            "White light is normally incident on a puddle of water (index of refraction 1.33). A thin (500 nm) layer of oil (index of refraction 1.5) floats on the surface of the puddle. Of the following, the most strongly reflected wavelength is	500 nm",
            "Which of the following is true about any system that undergoes a reversible thermodynamic process?	There are no changes in the internal energy of the system.",
            "The best type of laser with which to do spectroscopy over a range of visible wavelengths is a dye laser",
            "Excited states of the helium atom can be characterized as para- (antiparallel electron spins) and ortho- (parallel electron spins). The observation that an ortho- state has lower energy than the corresponding para- state can be understood in terms of which of the following?	The Heisenberg uncertainty principle",
            "The Barkhausen criterion for an oscillator	Loop gain should be unity",
            "Potentiometer method of DC voltage measurement is more accurate than direct measurement using a voltmeter because	It loads the circuit moderately.",
            "Which of these sets of logic gates are designated as universal gates?",
            "A single phase one pulse controlled circuit has a resistance R and counter emf E load 400 sin(314 t) as the source voltage. For a load counter emf of 200 V, the range of firing angle control is	30° to 150°.",
            "A box which tells the effect of inputs on control sub system is known as Data Box."
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
