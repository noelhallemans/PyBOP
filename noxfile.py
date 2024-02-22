import os
import nox


# nox options
nox.options.reuse_existing_virtualenvs = True
nox.options.venv_backend = "virtualenv"

# Environment variables to control CI behaviour for nox sessions
PYBOP_SCHEDULED = int(os.environ.get("PYBOP_SCHEDULED", 0))
PYBAMM_VERSION = os.environ.get("PYBAMM_VERSION", None)


@nox.session
def unit(session):
    session.install("-e", ".[all,dev]", silent=False)
    if PYBOP_SCHEDULED:
        session.run("pip", "install", f"pybamm=={PYBAMM_VERSION}", silent=False)
    session.run("pytest", "--unit", "-n", "auto")


@nox.session
def coverage(session):
    session.install("-e", ".[all,dev]", silent=False)
    if PYBOP_SCHEDULED:
        session.run("pip", "install", f"pybamm=={PYBAMM_VERSION}", silent=False)
    session.run(
        "pytest", "--unit", "--cov", "--cov-append", "--cov-report=xml", "-n", "auto"
    )
    session.run(
        "pytest",
        "--integration",
        "--cov",
        "--cov-append",
        "--cov-report=xml",
        "-n",
        "auto",
    )
    session.run("pytest", "--plots", "--cov", "--cov-append", "--cov-report=xml")


@nox.session
def integration(session):
    session.run_always("pip", "install", "-e", ".[all,dev]")
    session.install("pytest", "pytest-mock")
    session.run("pytest", "--integration", "-n", "auto")


@nox.session
def examples(session):
    session.run_always("pip", "install", "-e", ".[all,dev]")
    session.install("pytest", "pytest-mock")
    session.run("pytest", "--examples", "-n", "auto")


@nox.session
def notebooks(session):
    """Run the examples tests for Jupyter notebooks."""
    session.install("-e", ".[all,dev]", silent=False)
    if PYBOP_SCHEDULED:
        session.run("pip", "install", f"pybamm=={PYBAMM_VERSION}", silent=False)
    session.run(
        "pytest", "--notebooks", "--nbmake", "examples/", "-n", "auto", external=True
    )


@nox.session(name="tests")
def run_tests(session):
    """Run all the tests."""
    session.install("-e", ".[all,dev]", silent=False)
    if PYBOP_SCHEDULED:
        session.run("pip", "install", f"pybamm=={PYBAMM_VERSION}", silent=False)
    session.run(
        "pytest", "--unit", "--integration", "--nbmake", "--examples", "-n", "auto"
    )


@nox.session(name="doctest")
def run_doc_tests(session):
    """
    Checks if the documentation can be built, runs any doctests (currently not
    used).
    """
    session.install("-e", ".[all,docs,dev]", silent=False)
    session.run("pytest", "--docs")


@nox.session(name="pre-commit")
def lint(session):
    """
    Check all files against the defined pre-commit hooks.

    Credit: PyBaMM Team
    """
    session.install("pre-commit", silent=False)
    session.run("pre-commit", "run", "--all-files")


@nox.session(name="quick", reuse_venv=True)
def run_quick(session):
    """
    Run integration tests, unit tests, and doctests sequentially

    Credit: PyBaMM Team
    """
    run_tests(session)
    run_doc_tests(session)


@nox.session
def docs(session):
    """
    Build the documentation and load it in a browser tab, rebuilding on changes.
    Credit: PyBaMM Team
    """
    envbindir = session.bin
    session.install("-e", ".[all,docs]", silent=False)
    session.chdir("docs")
    # Local development
    if session.interactive:
        session.run(
            "sphinx-autobuild",
            "-j",
            "auto",
            "--open-browser",
            "-qT",
            ".",
            f"{envbindir}/../tmp/html",
        )
    # Runs in CI only, treating warnings as errors
    else:
        session.run(
            "sphinx-build",
            "-j",
            "auto",
            "-b",
            "html",
            "--keep-going",
            ".",
            "_build/html",
        )
