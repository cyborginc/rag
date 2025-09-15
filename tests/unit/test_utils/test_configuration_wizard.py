# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for configuration wizard functionality."""

import json
import os
import tempfile
from io import StringIO
from unittest.mock import patch, mock_open
from typing import Optional

import pytest
import yaml

from nvidia_rag.utils.configuration_wizard import (
    ConfigWizard,
    configclass,
    configfield,
    read_json_or_yaml,
    try_json_load,
    update_dict,
)


@configclass
class TestNestedConfig(ConfigWizard):
    """Test nested configuration class."""
    
    nested_field: str = configfield(
        "nested_field",
        default="nested_default",
        help_txt="A nested configuration field"
    )
    
    nested_int: int = configfield(
        "nested_int", 
        default=42,
        help_txt="A nested integer field"
    )


@configclass
class TestConfig(ConfigWizard):
    """Test configuration class for testing."""
    
    # Field with default value
    simple_field: str = configfield(
        "simple_field",
        default="default_value",
        help_txt="A simple test field"
    )
    
    # Field with custom environment variable name
    custom_env_field: str = configfield(
        "custom_env_field",
        default="custom_default",
        env_name="CUSTOM_ENV_VAR",
        help_txt="Field with custom environment variable"
    )
    
    # Field with env disabled
    no_env_field: str = configfield(
        "no_env_field",
        default="no_env_default",
        env=False,
        help_txt="Field without environment variable support"
    )
    
    # Integer field
    int_field: int = configfield(
        "int_field",
        default=123,
        help_txt="An integer field"
    )
    
    # Boolean field
    bool_field: bool = configfield(
        "bool_field",
        default=True,
        help_txt="A boolean field"
    )
    
    # Float field
    float_field: float = configfield(
        "float_field",
        default=3.14,
        help_txt="A float field"
    )
    
    # Nested configuration
    nested: TestNestedConfig = configfield(
        "nested",
        env=False,
        default=TestNestedConfig(),
        help_txt="Nested configuration"
    )


class TestConfigWizard:
    """Test cases for ConfigWizard class."""

    def test_basic_config_creation(self):
        """Test basic configuration creation with defaults."""
        config = TestConfig.from_dict({})
        
        assert config.simple_field == "default_value"
        assert config.custom_env_field == "custom_default" 
        assert config.no_env_field == "no_env_default"
        assert config.int_field == 123
        assert config.bool_field is True
        assert config.float_field == 3.14
        assert config.nested.nested_field == "nested_default"
        assert config.nested.nested_int == 42

    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        data = {
            "simpleField": "dict_value",
            "intField": 456,
            "boolField": False,
            "floatField": 2.71,
            "nested": {
                "nestedField": "dict_nested_value",
                "nestedInt": 100
            }
        }
        
        config = TestConfig.from_dict(data)
        
        assert config.simple_field == "dict_value"
        assert config.int_field == 456
        assert config.bool_field is False
        assert config.float_field == 2.71
        assert config.nested.nested_field == "dict_nested_value"
        assert config.nested.nested_int == 100

    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variables_auto_generated(self):
        """Test auto-generated environment variable names."""
        # Set environment variables using auto-generated names
        env_vars = {
            "APP_SIMPLEFIELD": "env_simple_value",
            "APP_INTFIELD": "999",
            "APP_BOOLFIELD": "false",
            "APP_FLOATFIELD": "1.23",
            "APP_NESTED_NESTEDFIELD": "env_nested_value",
            "APP_NESTED_NESTEDINT": "777"
        }
        
        with patch.dict(os.environ, env_vars):
            config = TestConfig.from_dict({})
            
            assert config.simple_field == "env_simple_value"
            assert config.int_field == 999
            assert config.bool_field is False
            assert config.float_field == 1.23
            assert config.nested.nested_field == "env_nested_value"
            assert config.nested.nested_int == 777

    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variables_custom_name(self):
        """Test custom environment variable names."""
        env_vars = {
            "CUSTOM_ENV_VAR": "custom_env_value"
        }
        
        with patch.dict(os.environ, env_vars):
            config = TestConfig.from_dict({})
            
            assert config.custom_env_field == "custom_env_value"

    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variables_json_parsing(self):
        """Test basic type conversion in environment variables."""
        env_vars = {
            "APP_SIMPLEFIELD": "env_string_value",  # String field gets string value
            "APP_INTFIELD": "42",  # Integer field gets parsed
            "APP_BOOLFIELD": "true",  # Boolean field gets parsed
            "APP_FLOATFIELD": "3.14159"  # Float field gets parsed
        }
        
        with patch.dict(os.environ, env_vars):
            config = TestConfig.from_dict({})
            
            assert config.simple_field == "env_string_value"
            assert config.int_field == 42
            assert config.bool_field is True
            assert config.float_field == 3.14159

    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variables_disabled(self):
        """Test that fields with env=False don't use environment variables."""
        env_vars = {
            "APP_NOENVFIELD": "should_not_be_used"
        }
        
        with patch.dict(os.environ, env_vars):
            config = TestConfig.from_dict({})
            
            # Should still use default value, not environment variable
            assert config.no_env_field == "no_env_default"

    def test_envvars_structure(self):
        """Test the structure of envvars return values."""
        envvars = TestConfig.envvars()
        
        # Find the custom env variable entry
        custom_env_entry = next(
            (env for env in envvars if env[0] == "CUSTOM_ENV_VAR"), None
        )
        
        assert custom_env_entry is not None
        assert custom_env_entry[1] == ("customEnvField",)  # JSON path
        assert custom_env_entry[2] == str  # Type

    def test_invalid_dict_data(self):
        """Test handling of invalid dictionary data."""
        with pytest.raises(RuntimeError, match="Configuration data is not a dictionary"):
            TestConfig.from_dict("not_a_dict")

    def test_none_dict_data(self):
        """Test handling of None dictionary data."""
        config = TestConfig.from_dict(None)
        
        # Should use defaults
        assert config.simple_field == "default_value"

class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_read_json_or_yaml_json(self):
        """Test reading JSON formatted data."""
        json_data = '{"key": "value", "number": 42}'
        stream = StringIO(json_data)
        
        result = read_json_or_yaml(stream)
        
        assert result == {"key": "value", "number": 42}

    def test_read_json_or_yaml_yaml(self):
        """Test reading YAML formatted data."""
        yaml_data = """
        key: value
        number: 42
        list:
          - item1
          - item2
        """
        stream = StringIO(yaml_data)
        
        result = read_json_or_yaml(stream)
        
        assert result["key"] == "value"
        assert result["number"] == 42
        assert result["list"] == ["item1", "item2"]

    def test_read_json_or_yaml_invalid(self):
        """Test reading invalid formatted data."""
        invalid_data = "invalid: json: yaml: {["
        stream = StringIO(invalid_data)
        
        with pytest.raises(ValueError) as exc_info:
            read_json_or_yaml(stream)
        
        # Should contain both JSON and YAML parser errors
        assert "JSON Parser Errors" in str(exc_info.value)
        assert "YAML Parser Errors" in str(exc_info.value)

    def test_read_json_or_yaml_non_seekable(self):
        """Test reading from non-seekable stream."""
        # Mock a non-seekable stream
        stream = StringIO('{"key": "value"}')
        stream.seekable = lambda: False
        
        with pytest.raises(ValueError, match="must be seekable"):
            read_json_or_yaml(stream)

    def test_try_json_load_valid_json(self):
        """Test JSON loading with valid JSON string."""
        result = try_json_load('{"key": "value"}')
        assert result == {"key": "value"}
        
        result = try_json_load('[1, 2, 3]')
        assert result == [1, 2, 3]
        
        result = try_json_load('42')
        assert result == 42
        
        result = try_json_load('true')
        assert result is True

    def test_try_json_load_invalid_json(self):
        """Test JSON loading with invalid JSON string."""
        result = try_json_load('invalid json')
        assert result == 'invalid json'  # Should return original string
        
        result = try_json_load('{"incomplete": }')
        assert result == '{"incomplete": }'

    def test_update_dict_simple(self):
        """Test updating dictionary with simple path."""
        data = {}
        update_dict(data, ("key",), "value")
        
        assert data == {"key": "value"}

    def test_update_dict_nested(self):
        """Test updating dictionary with nested path."""
        data = {}
        update_dict(data, ("level1", "level2", "key"), "value")
        
        expected = {
            "level1": {
                "level2": {
                    "key": "value"
                }
            }
        }
        assert data == expected

    def test_update_dict_existing_path(self):
        """Test updating dictionary with existing path."""
        data = {"level1": {"level2": {"existing": "old_value"}}}
        update_dict(data, ("level1", "level2", "key"), "new_value")
        
        expected = {
            "level1": {
                "level2": {
                    "existing": "old_value",
                    "key": "new_value"
                }
            }
        }
        assert data == expected

    def test_update_dict_overwrite_false(self):
        """Test not overwriting existing values when overwrite=False."""
        data = {"key": "existing_value"}
        update_dict(data, ("key",), "new_value", overwrite=False)
        
        # Should keep existing value
        assert data == {"key": "existing_value"}

    def test_update_dict_overwrite_true(self):
        """Test overwriting existing values when overwrite=True."""
        data = {"key": "existing_value"}
        update_dict(data, ("key",), "new_value", overwrite=True)
        
        # Should update to new value
        assert data == {"key": "new_value"}

    def test_update_dict_non_dict_intermediate(self):
        """Test handling of non-dict intermediate values."""
        data = {"level1": "not_a_dict"}
        update_dict(data, ("level1", "level2", "key"), "value")
        
        # Should not update when intermediate value is not a dict
        assert data == {"level1": "not_a_dict"}


class TestConfigField:
    """Test cases for configfield function."""

    def test_configfield_basic(self):
        """Test basic configfield creation."""
        field = configfield("test_field", help_txt="Test help")
        
        assert field.json.keys == ("testField",)  # camelCase conversion
        assert field.metadata["help"] == "Test help"
        assert field.metadata["env"] is True
        assert field.metadata["env_name"] is None

    def test_configfield_custom_env_name(self):
        """Test configfield with custom environment variable name."""
        field = configfield("test_field", env_name="CUSTOM_ENV", help_txt="Test help")
        
        assert field.metadata["env_name"] == "CUSTOM_ENV"

    def test_configfield_env_disabled(self):
        """Test configfield with environment variable disabled."""
        field = configfield("test_field", env=False, help_txt="Test help")
        
        assert field.metadata["env"] is False

    def test_configfield_invalid_name(self):
        """Test configfield with invalid name type."""
        with pytest.raises(TypeError, match="Provided name must be a string"):
            configfield(123, help_txt="Test help") 