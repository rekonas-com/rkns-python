import pytest

from rkns.util.misc import (
    apply_check_open_to_all_methods,
    check_open,
    import_from_string,
)


def test_import_from_string():
    """
    Tests that import_from_string correctly imports objects from strings.

    Test translated to pytest from
    https://github.com/django/django/blob/stable/5.1.x/tests/utils_tests/test_module_loading.py

    """

    cls = import_from_string("rkns.util.import_from_string")
    assert cls == import_from_string

    with pytest.raises(ImportError):
        import_from_string("no_dots_in_path")

    with pytest.raises(ModuleNotFoundError):
        import_from_string("utils_tests.unexistent")


# Mock class for testing
@apply_check_open_to_all_methods
class MockRKNS:
    def __init__(self):
        self._is_closed = False

    def close(self):
        self._is_closed = True

    def some_method(self):
        return "Method executed"


class MockRKNS2:
    def __init__(self):
        self._is_closed = False

    def close(self):
        self._is_closed = True

    @check_open
    def some_method(self):
        return "Method executed"


class TestCheckOpenDecorator:
    def test_check_open_decorator(self):
        mock_rkns = MockRKNS()
        assert mock_rkns.some_method() == "Method executed"

        mock_rkns.close()
        with pytest.raises(RuntimeError) as exc_info:
            mock_rkns.some_method()
        assert "Cannot execute some_method: RKNS object has been closed" in str(
            exc_info.value
        )

    def test_check_open_decorator2(self):
        mock_rkns = MockRKNS2()
        assert mock_rkns.some_method() == "Method executed"

        mock_rkns.close()
        with pytest.raises(RuntimeError) as exc_info:
            mock_rkns.some_method()
        assert "Cannot execute some_method: RKNS object has been closed" in str(
            exc_info.value
        )

    def test_apply_check_open_to_all_methods(self):
        mock_rkns = MockRKNS()
        assert mock_rkns.some_method() == "Method executed"

        mock_rkns.close()
        with pytest.raises(RuntimeError) as exc_info:
            mock_rkns.some_method()
        assert "Cannot execute some_method: RKNS object has been closed" in str(
            exc_info.value
        )

    def test_apply_check_open_to_all_methods_ignores_static_class_methods(self):
        @apply_check_open_to_all_methods
        class TestClass:
            @staticmethod
            def static_method():
                return "Static method"

            @classmethod
            def class_method(cls):
                return "Class method"

            def instance_method(self):
                return "Instance method"

        test_instance = TestClass()
        assert test_instance.static_method() == "Static method"
        assert test_instance.class_method() == "Class method"
        assert test_instance.instance_method() == "Instance method"
