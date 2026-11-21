import os
import platform
import sys

REPO = os.environ.get("EPITCR_REPO", "./epiTCR")
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


def imblearn_check():
    from imblearn.under_sampling import RandomUnderSampler  # noqa: F401
    return "importable"


def repo_path():
    if not os.path.isdir(REPO):
        raise FileNotFoundError(REPO)
    return os.path.abspath(REPO)


def model_file():
    p = "models_unzipped/rdforestWithoutMHCModel.pickle"
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return f"{os.path.getsize(p) / 1e6:.1f} MB"


def test_data():
    p = os.path.join(REPO, "data", "splitData", "withoutMHC", "test", "test01.csv")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    with open(p) as f:
        return f"{sum(1 for _ in f) - 1:,} rows"


print(f"python {sys.version.split()[0]} / {platform.system()} {platform.machine()}")

check("sklearn", version_of("sklearn"))
check("numpy", version_of("numpy"))
check("pandas", version_of("pandas"))
check("imblearn", imblearn_check)
check("repo", repo_path)
check("model", model_file)
check("test data", test_data)

try:
    import sklearn
    if sklearn.__version__ != "1.2.0":
        print(f"[WARN] sklearn {sklearn.__version__}, model pickle was built with 1.2.0")
except Exception:
    pass

if failures:
    print(f"\n{len(failures)} failed: {', '.join(failures)}")
    sys.exit(1)

print("\nok")
