# Ensures the repo root is on sys.path so `pytest` can import `agent`
# without installing the package (plain `pytest` doesn't add cwd like
# `python -m pytest` does).
