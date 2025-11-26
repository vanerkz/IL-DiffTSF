# IL-DiffTSF: Invertible Latent Diffusion for Probabilistic Time Series Forecasting 
This implementation is based on the paper:
V. K. Z. Koh, S. Lin, Y. Li, Z. Lin, and B. Wen, *"IL-DiffTSF: Invertible Latent Diffusion for Probabilistic Time Series Forecasting,"* IEEE Internet of Things Journal, 2025. DOI: [10.1109/JIOT.2025.3636872](https://doi.org/10.1109/JIOT.2025.3636872).

## Overview

Internet of Things (IoT) devices generate large volumes of time series data that are often volatile and complex, making probabilistic time series forecasting (TSF) essential for modeling the distribution of future outcomes. Recently, diffusion-based TSF methods have gained attention for their ability to learn complex distributions. However, they typically apply the diffusion process directly in the time domain, which may struggle to capture complex temporal dependencies, thus limiting the full potential of the diffusion process. Besides, they obtain probabilistic forecasts by sampling multiple plausible outcomes from the learned distribution, which is time-consuming and less effective. To solve these problems, we propose Invertible Latent Diffusion for probabilistic Time Series Forecasting (IL-DiffTSF), a novel approach based on a latent diffusion model. Specifically, we design an invertible latent projection between time series and latent space, where a conditional diffusion process is applied. This design ensures bidirectional consistency and minimal information loss, enabling more accurate TSF. Moreover, instead of sampling-based probabilistic forecasting, IL-DiffTSF represents uncertainties by directly learning a mapping from latent representations to prediction errors, achieving faster and more reliable uncertainty estimates. Experiments on univariate and multivariate benchmarks validate the efficiency and effectiveness of IL-DiffTSF.

## Training and Evaluation

To train and evaluate **IL-DiffTSF**, simply run the provided Jupyter notebook.

```bash
cd IL-DiffTSF
jupyter notebook run_IL_DiffTSF.ipynb
```
Inside the notebook, you can select the dataset by modifying the `value` variable (e.g., `"ETTh1"`, `"ETTm2"`, `"weather"`, `"exchange"`, etc.).

You can also control the forecasting type via the `--features` argument:

* `M` – multivariate input predicting multivariate output
* `S` – univariate input predicting univariate output

Other important arguments you may want to adjust for your experiments:

In addition to dataset and sequence parameters, IL-DiffTSF provides several configurable model and diffusion options:
* `--label_len` – length of the input (historical) sequence fed to the model
* `--pred_len` – length of the prediction (future) sequence
* `--batch_size` – number of samples per batch for training
* `--train_epochs` – number of training epochs
* `--d_model` – dimension of the model embeddings (default: 256)
* `--n_times` – number of diffusion steps (default: 100)
* `--sampling` – whether to generate multiple sampled outcomes for probabilistic forecasts (default: False)
* `--sampling_times` – number of outcomes to sample if `--sampling` is True (default: 30)
* `--train_est` – whether to train the estimator network (EST) (default: True)
* `--offset` – whether to apply an output offset (default: False)

These parameters let you control the **model complexity, diffusion process, and uncertainty estimation behavior** during training and evaluation.

## Citation

If you use IL-DiffTSF in your research, please cite:
```bibtex
@article{koh2025ildiffftsf,
  author={Koh, V. K. Z. and Lin, S. and Li, Y. and Lin, Z. and Wen, B.},
  title={IL-DiffTSF: Invertible Latent Diffusion for Probabilistic Time Series Forecasting},
  journal={IEEE Internet of Things Journal},
  year={2025},
  doi={10.1109/JIOT.2025.3636872},
  keywords={Probabilistic logic; Diffusion processes; Predictive models; Time-domain analysis; Time series analysis; Internet of Things; Training; Diffusion models; Accuracy; Limiting; Time series forecasting; latent diffusion model; invertible latent diffusion}
}














