from algo.pn_utils.maniskill_learn.utils.data import to_torch
from algo.pn_utils.maniskill_learn.utils.torch import ExtendedModule
from ..builder import POLICYNETWORKS, build_backbone, build_dense_head
from ..utils import replace_placeholder_with_args, get_kwargs_from_shape


@POLICYNETWORKS.register_module()
class ContinuousPolicy(ExtendedModule):
    def __init__(self, nn_cfg, policy_head_cfg, action_space, obs_shape=None, action_shape=None, encoder_cfg=None, if_contrast=False):
        super(ContinuousPolicy, self).__init__()
        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        nn_cfg = replace_placeholder_with_args(nn_cfg, **replaceable_kwargs)
        self.if_contrast = if_contrast #new
        if if_contrast:
            encoder_cfg = replace_placeholder_with_args(encoder_cfg, **replaceable_kwargs) #new
            self.encoder = build_backbone(encoder_cfg) #new
            self.if_contrast = if_contrast #new
        self.backbone = build_backbone(nn_cfg)

        if action_space.is_bounded():
            low = action_space.low
            high = action_space.high
            scale_prior = (high - low) / 2
            bias_prior = (low + high) / 2
            policy_head_cfg['scale_prior'] = scale_prior
            policy_head_cfg['bias_prior'] = bias_prior
        self.policy_head = build_dense_head(policy_head_cfg)

    def init_weights(self, pretrained=None, init_cfg=None):
        self.backbone.init_weights(pretrained, **init_cfg)

    def forward(self, state, num_actions=1, mode='sample', detach_encoder=False):
        if self.if_contrast: #new
            feature = self.encoder(state)
            if detach_encoder:
                feature = feature.detach()
            all_info = self.policy_head(self.backbone(feature), num_actions=num_actions)
        else:
            all_info = self.policy_head(self.backbone(state), num_actions=num_actions)
        sample, log_prob, mean = all_info[:3]
        if mode == 'all':
            return all_info
        elif mode == 'eval':
            return mean
        elif mode == 'sample':
            return sample
        else:
            raise ValueError(f"Unsupported mode {mode}!!")