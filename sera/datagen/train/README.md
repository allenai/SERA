# Training Instructions

We primarily use axolotl for training, but also validate with other frameworks such as llamafactory and unsloth. We provide all training configs that we use in train_config/. We do not include these frameworks in the dependencies of SERA and encourage users to install whichever framework they are must comfortable with.

We include bash scripts to run axolotl and unsloth training in `train_axolotl_8b.sh`, `train_axolotl_32b.sh`, and `train_unsloth.sh`.

## Note on Axolotl Training

Axolotl will add a `_checkpoint_wrapped_module` prefix to weight names in the state dict. We include `convert_axolotl_checkpoint.py` as a post-hoc adjustment to the final checkpoint. If this change is not applied, the trained model will not be compatible with vLLM or sgLang.

## Postprocessing

Some gotchas during training are, which we handle in this repository are:
- Making sure that the correct system prompt is added into the training data. axolotl does _not_ automatically apply chat templates, so we manually add the system prompt including the tools when we postprocess our data.
- Making sure that the model only trains on assistant turns (which we control using a train key we add in postprocessing)

If using alternative models, make sure that these requirements are satisfied.
