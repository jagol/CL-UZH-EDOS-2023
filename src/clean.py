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
    
    def __init__(self, cleaning_type: str, remove_brackets: bool = False, remove_emojis: bool = True) -> None:
        self.cleaning_type = cleaning_type
        self.remove_brackets = remove_brackets
        self.remove_emojis = remove_emojis
    
    def clean(self, text: str) -> str:
        ctext = re.sub(self.amp_regex, '&', text)  # sub wrong decoded &amp;
        if self.remove_brackets:
            ctext = re.sub(self.brackets_regex, '', ctext)  # remove stuff in brackets
        if self.remove_emojis:
            ctext = self._remove_emojis(ctext)
        if self.cleaning_type == 'remove':
            ctext = self._remove(ctext)
        elif self.cleaning_type == 'replace':
            ctext = self._replace(ctext)
        ctext = re.sub(self.white_space_regex, ' ', ctext) # remove unnecessary white-space
        return ctext

    def _remove(self, text: str) -> str:
        """Remove mentions and urls."""
        text = re.sub(self.usr_regex, '', text)
        text = re.sub(self.url_regex, '', text)
        return text

    def _replace(self, text: str) -> str:
        """Replace mentions and urls placeholders."""
        text = re.sub(self.usr_regex, '[USER]', text)
        text = re.sub(self.url_regex, '[URL]', text)
        return text
    
    @staticmethod
    def _remove_emojis(text: str) -> str:
        return emoji.get_emoji_regexp().sub(r'', text)