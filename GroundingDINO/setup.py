# setup.py - snippet to replace original get_extensions()

def get_extensions():
    # if target is jittor migration, skip building torch C++ extensions
    if os.environ.get("MIGRATE_TO_JITTOR", "") == "1":
        print("MIGRATE_TO_JITTOR=1 -> skip building torch extensions")
        return []

    # also, if jittor is present and torch not present, skip:
    try:
        import jittor  # noqa
        # If user intentionally installed jittor and want to skip torch ext:
        if os.environ.get("FORCE_TORCHEXT", "") != "1":
            print("Detected jittor in environment -> skip building torch extensions (set FORCE_TORCHEXT=1 to override)")
            return []
    except Exception:
        pass

    # --- original code below ---
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "groundingdino", "models", "GroundingDINO", "csrc")

    # ... rest of original get_extensions code unchanged ...
