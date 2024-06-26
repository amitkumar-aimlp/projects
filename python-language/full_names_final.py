# Fixing the code
# When a test fails, the code needs to be modified until the test
# passes again. (Don’t make the mistake of rewriting your tests to fit
# your new code.) Here we can make the middle name optional.
def get_full_name(first, last, middle=''):
    """Return a full name."""
    if middle:
        full_name = f"{first} {middle} {last}"
    else:
        full_name = f"{first} {last}"
    return full_name.title()