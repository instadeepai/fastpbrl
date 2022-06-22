from absl import logging as absl_logging

# Filter out logs from absl 'Unable to initialize backend 'tpu_driver'
absl_logging.set_verbosity(absl_logging.WARNING)
