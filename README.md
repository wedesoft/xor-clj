# xor-clj

Implementing XOR using a neural network (the hello world of machine learning).

* Install Java
* Install [Clojure](https://clojure.org/) (clj tool)
* Install [Python 3.13](https://www.python.org/)
* Install [uv](https://docs.astral.sh/uv/)

Install the Python packages using `uv sync`.
If your system comes with a newer version of Python, run `uv lock` first.
If your system comes with an older version or if you want to use a Torch version with GPU support, you need to edit the *project.toml* file before running `uv lock` and `uv sync`.

Make sure using `uv python list` that the uv environment is using the system installed Python executable otherwise it seems to fail to import the `_ctypes` module.

Run `uv run clj -M -m xor-clj.xor` to run XOR example.
