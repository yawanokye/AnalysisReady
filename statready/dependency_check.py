from __future__ import annotations


def check_runtime_dependencies() -> dict[str, str]:
    """Import every third-party runtime dependency used by StatReady.

    Render executes this during the Docker build so a missing dependency is
    reported before the web service is started.
    """
    import streamlit
    import pandas
    import numpy
    import scipy
    import statsmodels
    import sklearn
    import openpyxl
    import xlrd
    import docx
    import plotly
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot
    import PIL
    import networkx

    modules = {
        "streamlit": streamlit,
        "pandas": pandas,
        "numpy": numpy,
        "scipy": scipy,
        "statsmodels": statsmodels,
        "scikit-learn": sklearn,
        "openpyxl": openpyxl,
        "xlrd": xlrd,
        "python-docx": docx,
        "plotly": plotly,
        "matplotlib": matplotlib,
        "Pillow": PIL,
        "networkx": networkx,
    }
    return {name: str(getattr(module, "__version__", "installed")) for name, module in modules.items()}


if __name__ == "__main__":
    print(check_runtime_dependencies())
