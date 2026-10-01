"""Fail-closed seed handling in the fresh RLinf_support runtime only."""


def install_strict_seed_runtime():
    import sapien.core as sapien
    from robotwin.envs import vector_env as runtime

    if getattr(runtime.SubEnv, '_flux_strict_seeds', False):
        return

    class RoutedEngine(sapien.Engine):

        def create_scene(self, config=None):
            if config is not None:
                sapien.physx.set_scene_config(config)
            # CUDA ordinal is local to this Ray worker's visible device; do
            # not let Vulkan's auto-selection put both env workers on GPU 0.
            return sapien.Scene([
                sapien.physx.PhysxCpuSystem(),
                sapien.render.RenderSystem('cuda:0'),
            ])

    sapien.Engine = RoutedEngine

    class StrictSeedSubEnv(runtime.SubEnv):
        _flux_strict_seeds = True

        def setup_task(self):
            self.close()
            self.task = runtime.class_decorator(self.task_name)
            with self.global_lock, self.lock:
                task = runtime.class_decorator(self.task_name)
                try:
                    task.setup_demo(
                        now_ep_num=self.env_seed,
                        seed=self.env_seed,
                        is_test=True,
                        **self.args)
                    self.episode_info_list = [task.get_info()]
                finally:
                    task.close_env()

        def reset(self, env_seed=None):
            with self.global_lock, self.lock:
                if self.task is not None:
                    self.reset_count += 1
                    self.task.close_env(clear_cache=self.reset_count %
                                        self.clear_cache_freq == 0)
                if env_seed is not None:
                    self.env_seed = env_seed
                # Refresh metadata for this seed, not the slot's first scene.
                # Any unstable reset raises; never substitute seed+1.
                self.task.setup_demo(
                    now_ep_num=self.env_seed, seed=self.env_seed, **self.args)
                self.episode_info_list = [self.task.get_info()]
                # Upstream np.random.choice is global/thread dependent. Pick
                # deterministically for paired evaluations and record the seed.
                descriptions = runtime.generate_episode_descriptions(
                    self.task_name, self.episode_info_list, 1, self.env_seed)
                choices = descriptions[0][self.instruction_type]
                self.instruction = str(
                    runtime.np.random.default_rng(
                        self.env_seed).choice(choices))
                self.args['instruction'] = self.instruction
                # Unlike upstream, an unstable seed stops the experiment. It
                # must never silently become seed+1 in a held-out comparison.
                self.task.set_instruction(self.instruction)
                self.task.step_lim = self.args['step_lim']
                self.task.run_steps = 0
                self.task.reward_step = 0

    runtime.SubEnv = StrictSeedSubEnv
