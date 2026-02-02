from datetime import datetime, timezone


def get_time():
    """Get current time"""
    return datetime.now(timezone.utc)
