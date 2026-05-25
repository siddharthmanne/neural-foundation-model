"""Stub tokenizers used to verify the eval harness behaves as expected.

These are not training targets — they exist so test_tokenizer.py can assert
predictable harness behavior on inputs with known properties (uniform random
tokens should yield perplexity near V; constant tokens should yield mean run
length L; etc).
"""

from .tokenizers import ConstantTokenizer, InformativeStubTokenizer, RandomTokenizer

__all__ = ["ConstantTokenizer", "InformativeStubTokenizer", "RandomTokenizer"]
