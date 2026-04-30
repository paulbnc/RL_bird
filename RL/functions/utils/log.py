def _log(message, log_file, verbose=True):
    if verbose:
        print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")