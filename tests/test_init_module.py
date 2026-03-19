import importlib
import zipfile


def test_has_pharlap_lib(tmp_path):
    t = importlib.import_module("hfpytrace")
    p = tmp_path / "pharlap_lib"
    p.mkdir()
    assert not t._has_pharlap_lib(p)
    (p / "pharlap_4.5.3").mkdir()
    assert t._has_pharlap_lib(p)


def test_extract_pharlap_lib_from_archive(tmp_path):
    t = importlib.import_module("hfpytrace")
    archive = tmp_path / "trace.zip"
    dst = tmp_path / "out"

    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("trace-main/pharlap_lib/rt_2D.m", "x")
        zf.writestr("trace-main/pharlap_lib/startup_2D.m", "x")
        zf.writestr("trace-main/pharlap_lib/pharlap_4.5.3/dat/a.txt", "ok")

    t._extract_pharlap_lib_from_archive(archive, dst)
    assert (dst / "pharlap_4.5.3" / "dat" / "a.txt").exists()
    assert not (dst / "rt_2D.m").exists()


def test_ensure_pharlap_lib_skip(monkeypatch):
    t = importlib.import_module("hfpytrace")
    monkeypatch.setenv("HFPYTRACE_SKIP_PHARLAP_DOWNLOAD", "1")
    p = t.ensure_pharlap_lib()
    assert str(p).endswith("pharlap_lib")
