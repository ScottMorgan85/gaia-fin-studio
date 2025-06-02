import sys
import types
import importlib
import os
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_utils_with_stubs():
    # Minimal pandas stub
    pd = types.ModuleType('pandas')
    class DataFrame:
        def __init__(self, *args, **kwargs):
            pass
    pd.DataFrame = DataFrame
    pd.read_excel = lambda *args, **kwargs: DataFrame()
    pd.to_datetime = lambda x: x
    pd.Timestamp = lambda *args, **kwargs: None
    pd.NaT = None
    sys.modules.setdefault('pandas', pd)

    # Other lightweight stubs
    for name in ['streamlit', 'yfinance']:
        sys.modules.setdefault(name, types.ModuleType(name))

    # Plotly stubs
    plotly = types.ModuleType('plotly')
    plotly.graph_objects = types.ModuleType('plotly.graph_objects')
    plotly.express = types.ModuleType('plotly.express')
    sys.modules.setdefault('plotly', plotly)
    sys.modules.setdefault('plotly.graph_objects', plotly.graph_objects)
    sys.modules.setdefault('plotly.express', plotly.express)

    # Streamlit extras stub
    metric_cards = types.ModuleType('streamlit_extras.metric_cards')
    metric_cards.style_metric_cards = lambda *args, **kwargs: None
    sys.modules.setdefault('streamlit_extras.metric_cards', metric_cards)

    # Reportlab stubs
    reportlab = types.ModuleType('reportlab')
    reportlab.lib = types.ModuleType('reportlab.lib')
    reportlab.lib.pagesizes = types.ModuleType('reportlab.lib.pagesizes')
    reportlab.lib.pagesizes.letter = None
    reportlab.lib.styles = types.ModuleType('reportlab.lib.styles')
    reportlab.lib.styles.getSampleStyleSheet = lambda: None
    reportlab.platypus = types.ModuleType('reportlab.platypus')
    for attr in ['SimpleDocTemplate', 'Paragraph', 'Spacer', 'Image']:
        setattr(reportlab.platypus, attr, object)
    sys.modules.setdefault('reportlab', reportlab)
    sys.modules.setdefault('reportlab.lib', reportlab.lib)
    sys.modules.setdefault('reportlab.lib.pagesizes', reportlab.lib.pagesizes)
    sys.modules.setdefault('reportlab.lib.styles', reportlab.lib.styles)
    sys.modules.setdefault('reportlab.platypus', reportlab.platypus)

    # Assets package stubs
    assets_pkg = types.ModuleType('assets')
    collector = types.ModuleType('assets.Collector')
    collector.InfoCollector = object
    portfolio = types.ModuleType('assets.Portfolio')
    portfolio.Portfolio = object
    stock = types.ModuleType('assets.Stock')
    stock.Stock = object
    assets_pkg.Collector = collector
    assets_pkg.Portfolio = portfolio
    assets_pkg.Stock = stock
    sys.modules.setdefault('assets', assets_pkg)
    sys.modules.setdefault('assets.Collector', collector)
    sys.modules.setdefault('assets.Portfolio', portfolio)
    sys.modules.setdefault('assets.Stock', stock)

    # Groq stub with minimal interface
    groq_mod = types.ModuleType('groq')
    class DummyChat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                message = types.SimpleNamespace(content="")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])
    class DummyGroq:
        def __init__(self, *args, **kwargs):
            self.chat = DummyChat()
    groq_mod.Groq = DummyGroq
    sys.modules.setdefault('groq', groq_mod)

    return importlib.import_module('utils')


utils = load_utils_with_stubs()

def test_format_currency_float():
    assert utils.format_currency(1234.56) == "$1,234.56"


def test_format_currency_decimal():
    assert utils.format_currency(Decimal('1234.56')) == "$1,234.56"


def test_format_currency_zero():
    assert utils.format_currency(0) == "$0.00"


def test_format_currency_negative():
    assert utils.format_currency(-1234.56) == "-$1,234.56"
