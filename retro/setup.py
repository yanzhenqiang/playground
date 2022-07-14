from setuptools import setup, find_packages

setup(
  name = 'transformer_with_continuous_and_memory',
  packages = find_packages(exclude=[]),
  description = 'RETRO - Retrieval Enhanced Transformer - Pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention-mechanism',
    'retrieval',
  ],
  install_requires=[
    'autofaiss',
    'einops>=0.3',
    'numpy',
    'sentencepiece',
    'torch>=1.6',
    'tqdm'
  ],
)
