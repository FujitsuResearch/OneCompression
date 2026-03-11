"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from logging import getLogger, basicConfig


def setup_logger(
    level: str = "INFO", basic: str = "WARNING", filemode: str = "w", **kwargs
):
    """Setup the logger

    Args:
        level: Logging level for the 'onecomp' logger.
        basic: Logging level for the root logger used in `basicConfig`.
        **kwargs: Additional keyword arguments.
        logfile: Name of the log file to write logs to.
    """

    # Make keyword argument for the `basicConfig` function
    basic_config_args = {
        "level": basic,
        "format": "%(message)s",
    }
    if "logfile" in kwargs:
        basic_config_args["filename"] = kwargs["logfile"]
        basic_config_args["filemode"] = filemode

    basicConfig(**basic_config_args)
    getLogger("onecomp").setLevel(level)
