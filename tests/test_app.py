"""Exercise Streamlit rendering, navigation and submitted forms."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_navigation_and_individual():
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert not app.error
    for page in ["Clasificación individual", "Carga CSV", "Laboratorio", "Evaluación", "Inicio"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        assert not app.error, page
        if page == "Clasificación individual":
            assert len(app.number_input) == 30
            app.button[-1].click().run()
            assert app.error
            assert not any("Clase predicha:" in x.value for x in app.subheader)
            app.button[0].click().run()
            app.button[-1].click().run()
            assert not app.exception
            assert not app.error
            assert any("Clase predicha:" in x.value for x in app.subheader)
            app.button[2].click().run()
            assert all(x.value is None for x in app.number_input)
            assert not any("Clase predicha:" in x.value for x in app.subheader)
        if page == "Laboratorio":
            app.number_input[0].set_value(20.).run()
            app.button[-1].click().run()
            app.slider[0].set_value(.8).run()
            assert not app.exception
            assert not app.error
    app.sidebar.radio[0].set_value("Laboratorio").run()
    assert not app.exception
    assert all(x.value is not None for x in app.number_input)
