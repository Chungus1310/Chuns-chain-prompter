class OutputValidator:
    """
    Validates LLM output based on domain-specific rules.
    """
    DOMAIN_RULES = {
        "education": {
            "min_length": 150,
            "blacklist": ["XXX", "http://"],
            "required_terms": {"education", "learn"}
        },
        "marketing": {
            "min_length": 50,
            "cta_required": True,
            "blacklist": ["XXX", "http://"],
        },
        "data_analysis": {
            "min_length": 100,
            "blacklist": ["XXX", "http://"],
            "required_terms": {"data", "analysis"}
        }
    }

    @classmethod
    def validate(cls, text: str, domain: str) -> bool:
        """
        Validates the output text based on the specified domain's rules.
        Makes required terms validation optional and more flexible.
        """
        rules = cls.DOMAIN_RULES.get(domain, {})
        
        # Length validation
        if len(text) < rules.get("min_length", 100):
            return False
            
        # Blacklist validation
        if any(bad in text.lower() for bad in rules.get("blacklist", [])):
            return False
            
        # CTA validation
        if rules.get("cta_required", False) and not cls.has_cta(text):
            return False
            
        # Required terms validation - made optional and more flexible
        required_terms = rules.get("required_terms")
        if required_terms:
            # Check if at least 50% of required terms are present
            found_terms = sum(1 for term in required_terms if term in text.lower())
            if found_terms < len(required_terms) * 0.5:
                return False
                
        return True

    @staticmethod
    def has_cta(text: str) -> bool:
        """
        Checks if the text contains a call to action (CTA).

        Args:
            text: The text to check.

        Returns:
            True if a CTA is detected, False otherwise.
        """
        # Basic CTA detection (can be improved)
        cta_patterns = ["call to action", "learn more", "sign up", "buy now", "get started", "click here", "contact us"]
        return any(pattern in text.lower() for pattern in cta_patterns)