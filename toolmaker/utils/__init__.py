def remove_newlines(s: str) -> str:
    return s.replace("\n", " ").replace("\r", "")


def truncate_str(s: str, max_length: int = 15000) -> str:
    if len(s) <= max_length:
        return s

    return f"""NOTE: The following content had to be truncated because its length ({len(s)}) exceeds the maximum character length ({max_length}). Thus, the middle part of the content is omitted. Below is the content (between the <content> and </content> tags, with the truncated part replaced with <truncated/>):
<content>
{s[: max_length // 2]}
<truncated/>
{s[-max_length // 2 :]}
</content>
"""
