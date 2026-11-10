# SPDX-License-Identifier: Apache-2.0
"""HTTP status of each request failure."""

from onemeld.composition.types import (
    QueueFullError,
    UMMDeadlineError,
    UMMPlanningError,
    UMMRequestError,
)


def error_status_code(error: BaseException) -> int:
    """503 when full or out of time, 400 for a client error, 422 for an unplannable task."""
    if isinstance(error, (QueueFullError, UMMDeadlineError)):
        return 503
    elif isinstance(error, UMMRequestError):
        return 400
    elif isinstance(error, UMMPlanningError):
        return 422
    else:
        return 500
