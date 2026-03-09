# SPDX-License-Identifier: Apache-2.0
"""
Example of using SSD (Speculative Speculative Decoding) with vLLM.

SSD is a novel speculative decoding algorithm that parallelizes drafting
and verification by pre-speculating for multiple possible verification
outcomes in advance.

For more details, see: https://arxiv.org/pdf/2603.03251
"""

from vllm import LLM, SamplingParams


def main():
    # Configuration
    model_name = "meta-llama/Llama-3-8B-Instruct"
    draft_model_name = "meta-llama/Llama-3-1B-Instruct"
    
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate Fibonacci numbers.",
        "What are the benefits of machine learning?",
    ]

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=256,
    )

    print("=" * 80)
    print("Running SSD (Speculative Speculative Decoding) Example")
    print("=" * 80)
    
    print("\n1. Standard SSD (synchronous mode)")
    print("-" * 80)
    
    # Initialize LLM with SSD
    llm = LLM(
        model=model_name,
        speculative_config={
            "model": draft_model_name,
            "method": "ssd",
            "num_speculative_tokens": 6,
            "ssd_async": False,  # Synchronous mode
        },
    )

    # Generate text
    outputs = llm.generate(prompts, sampling_params)

    # Print outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")

    print("\n" + "=" * 80)
    print("2. Async SSD (tree cache enabled)")
    print("-" * 80)
    
    # Initialize LLM with Async SSD
    llm_async = LLM(
        model=model_name,
        speculative_config={
            "model": draft_model_name,
            "method": "ssd",
            "num_speculative_tokens": 7,
            "ssd_async": True,  # Asynchronous mode with tree cache
            "ssd_async_fan_out": 3,  # Fan-out factor
            "ssd_max_cached_sequences": 10000,
        },
    )

    # Generate text
    outputs_async = llm_async.generate(prompts, sampling_params)

    # Print outputs
    for output in outputs_async:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")

    print("\n" + "=" * 80)
    print("SSD Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
