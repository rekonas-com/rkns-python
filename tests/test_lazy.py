import numpy as np
import pytest
import zarr

from rkns.lazy import LazySignal


class TestLazySignal:
    @pytest.fixture
    def test_signal(self):
        """Create test signal with known scaling"""
        digital = zarr.array([[1000], [2000], [3000]], dtype="int16")
        return LazySignal.from_minmaxs(
            digital,
            pmin=np.array([[-1.0]]),
            pmax=np.array([[1.0]]),
            dmin=np.array([[0]]),
            dmax=np.array([[3000]]),
        )

    def test_shape_proxy(self, test_signal):
        """Test shape property delegates to source"""
        source = zarr.array(np.random.rand(10, 20))

        class DummyIndexer(LazySignal):
            def _transform(self, data_slice, idx):
                return data_slice * 2

        lazy = DummyIndexer(source, np.ones((1, 20)), np.ones((1, 20)))
        assert lazy.shape == (10, 20)

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
        assert arr.shape == (3, 1)
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
        signal = LazySignal.from_minmaxs(
            digital,
            pmin=-np.ones((1, 25)),
            pmax=np.ones((1, 25)),
            dmin=np.zeros((1, 25)),
            dmax=3000 * np.ones((1, 25)),
        )

        assert np.isscalar(signal[0, 0])
        assert signal[:10, :10].shape == (10, 10)
        assert signal[:10, :].shape == (10, digital.shape[1])
        assert signal[:, :].shape == signal[:].shape == (10, digital.shape[1])

        ref = signal._m * (digital + signal._bias)
        np.testing.assert_allclose(signal[:10, :2], ref[:10, :2])
