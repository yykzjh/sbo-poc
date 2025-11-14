from typing import Optional
from contextlib import contextmanager

import deep_gemm

@contextmanager
def configure_deep_gemm_num_sms(num_sms: Optional[int] = None):
    if num_sms is None:
        yield
    else:
        original_num_sms = deep_gemm.get_num_sms()
        deep_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            deep_gemm.set_num_sms(original_num_sms)
