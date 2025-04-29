import pytest
import chatglm

# existing tests remain here. Can modify to add additional binding tests if necessary

def test_bindings_available():
    # this is just a simple test to check if the bindings are working 
    assert chatglm.version() is not None
