import json
import os
from validation import OutputValidator

class OutputHandler:
    """
    Handles the output from the LLM, including formatting and validation.
    """
    def __init__(self):
        """
        Initializes the OutputHandler.
        """
        pass

    def format_output(self, output: str, format: str = "plain_text", domain: str = None) -> str:
        """
        Formats the output text based on the specified format and domain.

        Args:
            output: The output text from the LLM.
            format: The desired output format (plain_text, markdown, json, html).
            domain: The domain context.

        Returns:
            The formatted output text, or an error message if validation fails.
        """
        if not OutputValidator.validate(output, domain):
            return f"Error: Output failed domain validation for {domain}"
        
        if format == "plain_text":
            return output
        elif format == "markdown":
            # Add basic markdown formatting (e.g., headings, lists)
            return self._add_markdown_formatting(output)
        elif format == "json":
            # Try to convert the output to a JSON object
            try:
                return json.dumps(json.loads(output), indent=2)
            except json.JSONDecodeError:
                print("Warning: Output is not valid JSON. Returning as plain text.")
                return output
        elif format == "html":
            # Add basic HTML formatting (e.g., paragraphs, headings)
            return self._add_html_formatting(output)
        else:
            print(f"Warning: Unknown output format '{format}'. Returning as plain text.")
            return output

    def _add_markdown_formatting(self, output: str) -> str:
        """
        Adds basic markdown formatting to the output text.

        Args:
            output: The output text.

        Returns:
            The markdown-formatted output text.
        """
        # Add basic markdown formatting here (e.g., headings, lists)
        # This is a placeholder for a more advanced implementation
        return output

    def _add_html_formatting(self, output: str) -> str:
        """
        Adds basic HTML formatting to the output text.

        Args:
            output: The output text.

        Returns:
            The HTML-formatted output text.
        """
        # Add basic HTML formatting here (e.g., paragraphs, headings)
        # This is a placeholder for a more advanced implementation
        return f"<p>{output.replace(os.linesep, '<br>')}</p>"

    def present_output(self, output: str):
        """
        Presents the output to the user (currently prints to console).

        Args:
            output: The output text.
        """
        print(output)