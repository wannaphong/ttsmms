import pytest
from ttsmms import TTS, download

@pytest.fixture(scope="module")
def tts_instance():
    # Download the model and create a TTS instance
    dir_path = download("eng", "./data")
    return TTS(dir_path)

def test_tts_synthesis(tts_instance):
    # Test if TTS synthesis output is in the correct format
    text = "hello"
    result = tts_instance.synthesis(text)
    
    # Check if result is a dictionary
    assert isinstance(result, dict), "Output should be a dictionary"
    
    # Check if 'audio_bytes' key exists and is of type bytes
    assert 'audio_bytes' in result, "'audio_bytes' key should be in the output dictionary"
    assert isinstance(result['audio_bytes'], bytes), "'audio_bytes' should be of type bytes"
    
    # Check if 'sampling_rate' key exists and is of type int
    assert 'sampling_rate' in result, "'sampling_rate' key should be in the output dictionary"
    assert isinstance(result['sampling_rate'], int), "'sampling_rate' should be of type int"
    
    # Additional checks can be added as necessary
    # Example: Ensure the sampling rate is a specific value
    assert result['sampling_rate'] == 16000, "The sampling rate should be 16000"

# If additional edge cases or scenarios need to be tested, add more test functions
def test_tts_synthesis_empty_string(tts_instance):
    # Test synthesis with an empty string
    text = ""
    result = tts_instance.synthesis(text)
    
    # Check if result is a dictionary
    assert isinstance(result, dict), "Output should be a dictionary"
    
    # Check if 'audio_bytes' key exists and is of type bytes
    assert 'audio_bytes' in result, "'audio_bytes' key should be in the output dictionary"
    assert isinstance(result['audio_bytes'], bytes), "'audio_bytes' should be of type bytes"
    
    # Check if 'sampling_rate' key exists and is of type int
    assert 'sampling_rate' in result, "'sampling_rate' key should be in the output dictionary"
    assert isinstance(result['sampling_rate'], int), "'sampling_rate' should be of type int"

def test_tts_synthesis_long_text(tts_instance):
    # Test synthesis with a long text
    text = "hello " * 1000
    result = tts_instance.synthesis(text)
    
    # Check if result is a dictionary
    assert isinstance(result, dict), "Output should be a dictionary"
    
    # Check if 'audio_bytes' key exists and is of type bytes
    assert 'audio_bytes' in result, "'audio_bytes' key should be in the output dictionary"
    assert isinstance(result['audio_bytes'], bytes), "'audio_bytes' should be of type bytes"
    
    # Check if 'sampling_rate' key exists and is of type int
    assert 'sampling_rate' in result, "'sampling_rate' key should be in the output dictionary"
    assert isinstance(result['sampling_rate'], int), "'sampling_rate' should be of type int"

