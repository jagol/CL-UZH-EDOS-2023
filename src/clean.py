import emoji
import re


class Cleaner:
    usr_regex = re.compile(r'@\w+\b')
    url_regex = re.compile("https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:"
                           "[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)")
    white_space_regex = re.compile(r'\s+')
    amp_regex = re.compile(r'&amp;')
    lower_score_regex = re.compile(r'_')
    brackets_regex = re.compile(r'\[.*?\]')
    
    def __init__(self, cleaning_type: str) -> None:
        self.cleaning_type = cleaning_type
    
    def clean(self, text: str) -> str:
        ctext = re.sub(self.amp_regex, '&', text)  # sub wrong decoded &amp;
        if self.cleaning_type == 'remove':
            ctext = self._remove(ctext)
        elif self.cleaning_type == 'replace':
            ctext = self._replace(ctext)
        ctext = re.sub(self.white_space_regex, ' ', ctext) # remove unnecessary white-space
        return ctext

    def _remove(self, text: str) -> str:
        """Remove mentions, urls and emojis."""
        text = re.sub(self.usr_regex, '', text)
        text = re.sub(self.url_regex, '', text)
        text = emoji.replace_emoji(text, '')
        return text

    def _replace(self, text: str) -> str:
        """Replace mentions, urls and emojis with placeholders."""
        text = re.sub(self.usr_regex, '[USER]', text)
        text = re.sub(self.url_regex, '[URL]', text)
        text = emoji.demojize(text, delimiters=('[', ']'), language='en')
        return text
