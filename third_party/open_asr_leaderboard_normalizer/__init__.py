# Vendored from https://github.com/huggingface/open_asr_leaderboard/tree/main/normalizer (Apache-2.0, see LICENSE).
# Only the English normalizer is kept; the upstream __init__ also exported Armenian/multilingual helpers.
from .normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer
