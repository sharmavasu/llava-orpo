# src/constants.py
DEFAULT_IMAGE_TOKEN = "<image>"
# You might find LLaVA uses a specific index like -200 for images internally,
# or it relies on tokenizer_image_token to handle it.
# For now, the string placeholder is fine for tokenizer_image_token.