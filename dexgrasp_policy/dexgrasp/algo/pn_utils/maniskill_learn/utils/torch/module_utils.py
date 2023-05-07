from torch.nn import Module
from torch.nn.parallel import DataParallel


class CustomDataParallel(DataParallel):
    def __init__(self, *args, **kwargs):
        super(CustomDataParallel, self).__init__(*args, **kwargs)

    def __getitem__(self, name):
        return getattr(self.module, name)

    @property
    def device(self):
        return self.device_ids[0] if self.device_ids is not None else None


class ExtendedModule(Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __getitem__(self, name):
        return getattr(self, name)


class BaseAgent(ExtendedModule):
    def __init__(self):
        super(BaseAgent, self).__init__()
        self._device_ids = None
        self._output_device = None
        self._dim = None
        self._be_data_parallel = False

    def to_data_parallel(self, device_ids=None, output_device=None, axis=0):
        self._device_ids = device_ids
        self._output_device = output_device
        self._dim = axis
        self.recover_data_parallel()

    def to_normal(self):
        if self._be_data_parallel and self._device_ids is not None:
            self._be_data_parallel = False
            for module_name in dir(self):
                item = getattr(self, module_name)
                if isinstance(item, CustomDataParallel):
                    setattr(self, module_name, item.module)

    def recover_data_parallel(self):
        if self._device_ids is None:
            return
        self._be_data_parallel = True
        for module_name in dir(self):
            item = getattr(self, module_name)
            if isinstance(item, Module):
                setattr(self, module_name, CustomDataParallel(module=item, device_ids=self._device_ids,
                                                              output_device=self._output_device, dim=self._dim))
    def is_data_parallel(self):
        return self._be_data_parallel

    @property
    def device(self):
        if self._device_ids is None or not self._be_data_parallel:
            return next(self.parameters()).device
        else:
            return self._device_ids[0]

    def forward(self, obs, **kwargs):
        from algo.pn_utils.maniskill_learn.utils.data import to_torch
        obs = to_torch(obs, dtype='float32', device=self.device)
        return self.policy(obs, **kwargs)
