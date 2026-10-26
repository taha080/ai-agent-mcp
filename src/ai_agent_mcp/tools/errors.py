"""Shared error type for failures a model can recover from."""


class ToolFailure(Exception):
    """An expected tool failure with a message safe to show the model.

    Subclasses are raised by tool implementations. The MCP adapter turns them
    into MCP tool errors so the model sees ``is_error`` instead of a crash.
    """
