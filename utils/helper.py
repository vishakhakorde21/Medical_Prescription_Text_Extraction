def read_prompt(prompt_path):
    """
    Reads the prompt from a text file.

    Parameters:
        prompt_path (str): Path of the prompt file.

    Returns:
        str: Prompt text.
    """

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()

    return prompt