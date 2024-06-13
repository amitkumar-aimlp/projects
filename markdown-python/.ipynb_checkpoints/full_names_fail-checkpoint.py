# Modifying the function
# We’ll modify get_full_name() so it handles middle names, but
# we’ll do it in a way that breaks existing behavior.
def get_full_name(first, middle, last):
    """Return a full name."""
    full_name = f"{first} {middle} {last}"
    return full_name.title()