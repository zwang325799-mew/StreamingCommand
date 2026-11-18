from .paths import model_path, worker_path
"""One owned protocol process; the continuity view never owns a second engine."""
from .asr_backend import NativeASR, WHISPER


class SharedContinuityASR(NativeASR):
    def __init__(self, owner, original):
        super().__init__(device=owner.device,model_dir=original.model_dir,
                         context=original.context,cantonese_final_language=None,warmup=False)
        self.owner=owner
        self.worker=owner.worker
        self.status.update(shared_engine=True,model_variant='base',
                           shared_owner_model_dir=str(owner.model_dir))

    async def start(self):
        if self.owner._process is None or not self.owner.status.get('worker_ready'):
            raise RuntimeError('Shared owner is not loaded')
        self._closing=False
        self.status.update(worker_ready=self.owner.status['worker_ready'],
                           log_path=self.owner.status['log_path'],
                           shared_worker_pid=self.owner._process.pid)
        self._phase('ready')
        return self.status['worker_ready']

    async def request(self,op,**kwargs):
        if self._closing:raise RuntimeError('Shared view is closed')
        return await NativeASR.request(self.owner,op,model_variant='base',**kwargs)

    async def close(self):
        self._closing=True
        self._phase('closed')


def configure_shared_engine(asr, *, cache_mib=512, max_model_len=4096, bank_mode='swap'):
    original=asr._continuity_asr
    if asr._process is not None or original._process is not None:
        raise RuntimeError('Configure shared engine before starting models')
    if isinstance(original,SharedContinuityASR):
        raise RuntimeError('Shared ASR is already configured')
    asr.alternate_model_dir=original.model_dir
    asr.worker=worker_path("qwen_native_shared_continuity_worker.py")
    asr.kv_cache_memory_bytes=cache_mib*2**20
    asr.max_model_len=max_model_len
    asr.weight_bank_mode=bank_mode
    asr._continuity_asr=SharedContinuityASR(asr,original)
    asr.status.update(worker=str(asr.worker),asr_engine_count=1,
                      memory_policy='shared_exact_fp16_banks',
                      kv_cache_memory_bytes=asr.kv_cache_memory_bytes,
                      max_model_len=max_model_len,weight_bank_mode=bank_mode,quantization=None)
    return asr
