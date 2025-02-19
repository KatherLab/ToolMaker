def escape_latex[T: str | None](text: T) -> T:
    """
    Escape special LaTeX characters in a string.

    Args:
        text: The input text to escape

    Returns:
        The escaped text safe for LaTeX
    """
    if text is None:
        return text

    latex_special_chars = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
        "\\": "\\textbackslash{}",
        "<": "\\textless{}",
        ">": "\\textgreater{}",
        "`": "\\textasciigrave{}",
    }

    # First handle backslash separately since it may be part of other escape sequences
    text = text.replace("\\", "\\textbackslash{}")

    # Then handle the rest
    for char, replacement in latex_special_chars.items():
        if char != "\\":  # Skip backslash as we already handled it
            text = text.replace(char, replacement)

    return text
