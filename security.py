import bleach
import re

class InputSanitizer:
    """
    Provides methods for sanitizing user input to mitigate security risks.
    """
    @staticmethod
    def sanitize(input_text: str) -> str:
        """
        Sanitizes the input text by removing potentially harmful HTML tags and attributes,
        and allowing only a specific set of safe characters.

        Args:
            input_text: The input text to sanitize.

        Returns:
            The sanitized text.
        """
        cleaned = bleach.clean(input_text, tags=[], attributes={}, strip=True)
        return re.sub(r'[^\w\s\-_@.,!?()]', '', cleaned)[:5000]