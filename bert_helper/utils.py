
import re


def name_filter(name):
    """
    filter out not name string using the length, and whether containing the [UNK] token
    """
    if len(name) < 2:
        return False
    if "[UNK]" in name:
        return False
    return True


def get_sentences(content, name, min_window=5, max_window=25):
    """
    extract nearby sentance container NER
    """
    symbols = {"，", "。", "！", "？", ",", "!", "?"}
    results = []
    for match in re.finditer(name, content):
        span = match.span()
        # right
        right = len(content)
        for i in range(span[1]+min_window, min(len(content), span[1]+max_window)):
            if content[i] in symbols:
                right = i
                break
            right = i
        # lefg
        left = 0
        for i in reversed(range(max(0, span[0]-max_window), span[0]-min_window)):
            if content[i] in symbols:
                left = i+1
                break
            left = i+1
        results.append(content[left:right])
    return results


def content_split(article: str, max_len=488) -> list:
    "Split the article into multiple paragraphs"
    symbols = {"。", "，", ",", "？", "！"}
    contents = []
    remaining = article
    while len(remaining) > max_len:
        target = max_len + 1
        for i, char in enumerate(reversed(remaining[:max_len])):
            if char in symbols:
                target = max_len-i-1
                break
        contents.append(remaining[:target] + "。")
        remaining = remaining[(1+target):]
    if len(remaining) > 3:
        for i, char in enumerate(article[-max_len:]):
            if char in symbols:
                contents.append(article[-(max_len-i-1):])
                return contents
        contents.append(article[-(max_len-1):])
    return contents
