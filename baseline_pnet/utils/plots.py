"""
No-op plotting stubs.

The original `utils/plots.py` was removed during the slim port because the
user said they'll generate AUROC/AUPR figures themselves from the per-patient
`pred_score` columns dumped by the pipelines. Several pipeline modules still
`from utils.plots import …`, so this stub keeps imports working while making
plotting a no-op.

If you ever want figures emitted directly by P-NET, restore the original
plotting code (it's in the upstream repo:
https://github.com/marakeby/pnet_prostate_paper/blob/master/utils/plots.py).
"""
import logging


def _noop(*args, **kwargs):
    """Stub used by every plotting function in this module."""
    logging.debug("utils.plots stub called: %s; skipping (no-op)", _noop.__name__)


# Symbols imported elsewhere — every one is wired to the no-op stub.
generate_plots = _noop
plot_roc = _noop
plot_prc = _noop
plot_box_plot = _noop
plot_confusion_matrix = _noop
save_confusion_matrix = _noop
