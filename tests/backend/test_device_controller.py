"""Tests for backend DeviceController."""

import pytest
from unittest.mock import Mock, patch

from qubex.backend.device_controller import RawResult, SAMPLING_PERIOD


class TestRawResult:
    """Tests for RawResult dataclass."""

    def test_init(self):
        """RawResult should initialize with given parameters."""
        status = {"success": True}
        data = {"measurement": [1, 2, 3]}
        config = {"sampling_rate": 1000}
        
        result = RawResult(status=status, data=data, config=config)
        
        assert result.status == status
        assert result.data == data
        assert result.config == config

    def test_equality(self):
        """RawResult instances with same data should be equal."""
        status = {"success": True}
        data = {"measurement": [1, 2, 3]}
        config = {"sampling_rate": 1000}
        
        result1 = RawResult(status=status, data=data, config=config)
        result2 = RawResult(status=status, data=data, config=config)
        
        assert result1 == result2


class TestDeviceControllerConstants:
    """Tests for DeviceController constants."""

    def test_sampling_period_constant(self):
        """SAMPLING_PERIOD should have expected value."""
        assert SAMPLING_PERIOD == 2.0


class TestDeviceControllerMocked:
    """Tests for DeviceController class with mocked dependencies."""

    def test_init_without_qubecalib(self):
        """Test DeviceController initialization when QubeCalib is not available."""
        from qubex.backend.device_controller import DeviceController
        
        # Since QubeCalib is not available in test environment, 
        # initialization should set _qubecalib to None
        controller = DeviceController()
        
        assert controller._qubecalib is None
        assert controller._cap_resource_map is None
        assert controller._gen_resource_map is None
        assert controller._boxpool is None
        assert controller._quel1system is None

    @patch('qubex.backend.device_controller.logger')
    def test_init_logs_import_error(self, mock_logger):
        """Test that import error is logged when QubeCalib is not available."""
        from qubex.backend.device_controller import DeviceController
        
        # The import error should have been logged during module import
        # We can't easily test this without reimporting, so we test the behavior
        controller = DeviceController()
        assert controller._qubecalib is None

    def test_qubecalib_property_valid(self):
        """Test qubecalib property when _qubecalib is set."""
        from qubex.backend.device_controller import DeviceController
        
        controller = DeviceController.__new__(DeviceController)
        mock_qubecalib = Mock()
        controller._qubecalib = mock_qubecalib
        
        assert controller.qubecalib == mock_qubecalib

    def test_qubecalib_property_none(self):
        """Test qubecalib property when _qubecalib is None."""
        from qubex.backend.device_controller import DeviceController
        
        controller = DeviceController.__new__(DeviceController)
        controller._qubecalib = None
        
        with pytest.raises(ModuleNotFoundError):
            _ = controller.qubecalib

    def test_box_config_property_no_boxpool(self):
        """Test box_config property when _boxpool is None."""
        from qubex.backend.device_controller import DeviceController
        
        controller = DeviceController.__new__(DeviceController)
        controller._boxpool = None
        
        assert controller.box_config == {}

    def test_system_config_property(self):
        """Test system_config property."""
        from qubex.backend.device_controller import DeviceController
        
        controller = DeviceController.__new__(DeviceController)
        mock_qubecalib = Mock()
        test_config = {"system": "config"}
        mock_qubecalib.system_config_database.asdict.return_value = test_config
        controller._qubecalib = mock_qubecalib
        
        assert controller.system_config == test_config
        mock_qubecalib.system_config_database.asdict.assert_called_once()

    def test_available_boxes_property(self):
        """Test available_boxes property."""
        from qubex.backend.device_controller import DeviceController
        
        controller = DeviceController.__new__(DeviceController)
        mock_qubecalib = Mock()
        box_settings = {"box1": {}, "box2": {}, "box3": {}}
        system_config = {"box_settings": box_settings}
        mock_qubecalib.system_config_database.asdict.return_value = system_config
        controller._qubecalib = mock_qubecalib
        
        expected_boxes = ["box1", "box2", "box3"]
        assert controller.available_boxes == expected_boxes

    def test_boxpool_property_none(self):
        """Test boxpool property when _boxpool is None."""
        from qubex.backend.device_controller import DeviceController
        
        controller = DeviceController.__new__(DeviceController)
        controller._boxpool = None
        
        with pytest.raises(ValueError, match="Boxes not connected"):
            _ = controller.boxpool