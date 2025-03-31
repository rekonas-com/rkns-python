import numpy as np
import pytest
import zarr

from rkns.lazy import LazyIndexer, LazySignal


class TestLazyIndexer:
    def test_abstract_base_raises(self):
        """Test base class requires implementation"""
        with pytest.raises(NotImplementedError):

            class ConcreteIndexer(LazyIndexer):
                pass  # Missing _transform

            ConcreteIndexer(np.array([1, 2, 3]))[0]

    def test_shape_proxy(self):
        """Test shape property delegates to source"""
        source = zarr.array(np.random.rand(10, 20))

        class DummyIndexer(LazyIndexer):
            def _transform(self, chunk):
                return chunk * 2

        lazy = DummyIndexer(source)
        assert lazy.shape == (10, 20)


class TestLazySignal:
    @pytest.fixture
    def test_signal(self):
        """Create test signal with known scaling"""
        digital = zarr.array([1000, 2000, 3000], dtype="int16")
        return LazySignal(digital, pmin=-1.0, pmax=1.0, dmin=0, dmax=3000)

    def test_scaling_calculation(self, test_signal):
        """Verify EDF scaling formula"""
        assert pytest.approx(test_signal._m) == 2 / 3000
        assert pytest.approx(test_signal._bias) == -1500

    def test_value_transform(self, test_signal):
        """Test physical value conversion"""
        # Known test case:
        # physical = (2/3000) * (digital - 1500)
        assert pytest.approx(test_signal[0]) == -0.3333333
        assert pytest.approx(test_signal[2]) == 1.0

    def test_array_interface(self, test_signal):
        """Test NumPy compatibility"""
        arr = np.array(test_signal)
        assert arr.shape == (3,)
        assert arr.dtype == np.float64
        assert pytest.approx(arr[0]) == -0.333333  # Midpoint should be zero

    def test_slicing(self, test_signal):
        """Test partial materialization"""
        chunk = test_signal[1:3]
        assert len(chunk) == 2
        assert pytest.approx(chunk[1]) == 1.0

    def test_multiindex_slicing(self, test_signal):
        """Test partial materialization"""

        digital = zarr.array(np.arange(250).reshape(10, 25), dtype="int16")
        signal = LazySignal(digital, pmin=-1.0, pmax=1.0, dmin=0, dmax=3000)

        assert np.isscalar(signal[0, 0])
        assert signal[:10, :10].shape == (10, 10)
        assert signal[:10, :].shape == (10, digital.shape[1])

        # assert pytest.approx(chunk[1]) == 1.0
