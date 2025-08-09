# Greeter component for simple greetings
class GreeterComponent:
    display_name = "Greeter"
    description = "Creates a greeting for a given name, optionally excited."

    def build(self, name: str, excited: bool = True) -> str:
        # input validation
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        # core logic
        greeting = f"Hello, {name.strip()}"
        if excited:
            greeting += "!"
        else:
            greeting += "."
        return greeting