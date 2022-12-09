import os

from pytorch_lightning.callbacks import BasePredictionWriter
from util import save_as_images, save_as_np


class ImageWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        compare=False,
        n_steps=None,
        eval_mode="sample",
        conditional=True,
        sample_prefix="",
        save_vae=False,
        save_mode="image",
        is_norm=True,
    ):
        super().__init__(write_interval)
        assert eval_mode in ["sample", "recons"]
        self.output_dir = output_dir
        self.compare = compare
        self.n_steps = 1000 if n_steps is None else n_steps
        self.eval_mode = eval_mode
        self.conditional = conditional
        self.sample_prefix = sample_prefix
        self.save_vae = save_vae
        self.is_norm = is_norm
        self.save_fn = save_as_images if save_mode == "image" else save_as_np

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        rank = pl_module.global_rank
        if self.conditional:
            ddpm_samples_dict, vae_samples = prediction

            if self.save_vae:
                vae_samples = vae_samples.cpu()
                vae_save_path = os.path.join(self.output_dir, "vae")
                os.makedirs(vae_save_path, exist_ok=True)
                self.save_fn(
                    vae_samples,
                    file_name=os.path.join(
                        vae_save_path,
                        f"output_vae_{self.sample_prefix}_{rank}_{batch_idx}",
                    ),
                    denorm=self.is_norm,
                )
        else:
            ddpm_samples_dict = prediction

        # Write output images
        # NOTE: We need to use gpu rank during saving to prevent
        # processes from overwriting images
        for k, ddpm_samples in ddpm_samples_dict.items():
            ddpm_samples = ddpm_samples.cpu()

            # Setup dirs
            base_save_path = os.path.join(self.output_dir, k)
            img_save_path = os.path.join(base_save_path, "images")
            os.makedirs(img_save_path, exist_ok=True)

            # Save
            self.save_fn(
                ddpm_samples,
                file_name=os.path.join(
                    img_save_path, f"output_{self.sample_prefix }_{rank}_{batch_idx}"
                ),
                denorm=self.is_norm,
            )

        # FIXME: This is currently broken. Separate this from the core logic
        # into a new function. Uncomment when ready!
        # if self.compare:
        #     # Save comparisons
        #     (_, img_samples), _ = batch
        #     img_samples = normalize(img_samples).cpu()
        #     iter_ = vae_samples if self.eval_mode == "sample" else img_samples
        #     for idx, (ddpm_pred, pred) in enumerate(zip(ddpm_samples, iter_)):
        #         samples = {
        #             "VAE" if self.eval_mode == "sample" else "Original": pred,
        #             "DDPM": ddpm_pred,
        #         }
        #         compare_samples(
        #             samples,
        #             save_path=os.path.join(
        #                 self.comp_save_path, f"compare_form1_{rank}_{idx}.png"
        #             ),
        #         )
