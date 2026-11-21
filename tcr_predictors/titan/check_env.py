import os
import platform
import sys

REPO = os.environ.get("TITAN_REPO", "./TITAN")
failures = []


def check(label, fn):
    try:
        detail = fn()
        print(f"[OK]   {label}" + (f" -> {detail}" if detail else ""))
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")
        failures.append(label)


def version_of(mod):
    return lambda: getattr(__import__(mod), "__version__", "installed")


def paccmann():
    from paccmann_predictor.models import MODEL_FACTORY  # noqa: F401
    return "importable"


def pytoda_check():
    from pytoda.datasets import DrugAffinityDataset  # noqa: F401
    return "importable"


def repo_path():
    if not os.path.isdir(REPO):
        raise FileNotFoundError(REPO)
    return os.path.abspath(REPO)


def weights():
    p = os.path.join(REPO, "trained_model", "weights",
                     "best_ROC-AUC_bimodal_mca.pt")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return f"{os.path.getsize(p) / 1e6:.1f} MB"


def test_data():
    p = os.path.join(REPO, "datasets", "trained_model_test.csv")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    with open(p) as f:
        return f"{sum(1 for _ in f) - 1:,} rows"


def results_dir():
    p = os.path.join(REPO, "trained_model", "results")
    os.makedirs(p, exist_ok=True)
    return "ready"


print(f"python {sys.version.split()[0]} / {platform.system()} {platform.machine()}")

if sys.version_info[:2] != (3, 9):
    print(f"[WARN] python {sys.version_info.major}.{sys.version_info.minor}, expected 3.9")

if platform.system() == "Darwin" and platform.machine() == "arm64":
    print("[WARN] arm64 interpreter, expected x86_64 under Rosetta")

check("torch", version_of("torch"))
check("numpy", version_of("numpy"))
check("pandas", version_of("pandas"))
check("sklearn", version_of("sklearn"))
check("rdkit", version_of("rdkit"))
check("numba", version_of("numba"))
check("Bio", version_of("Bio"))
check("pytoda", pytoda_check)
check("paccmann_predictor", paccmann)
check("repo", repo_path)
check("weights", weights)
check("test data", test_data)
check("results dir", results_dir)

if failures:
    print(f"\n{len(failures)} failed: {', '.join(failures)}")
    sys.exit(1)

print("\nok")
