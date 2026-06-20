# hf-cache-preloader
Preloads Hugging Face model repositories onto Kubernetes node-local disk using a YAML manifest, node-targeted DaemonSet, and upstream proxy support, so vLLM and other inference workloads start from a warm local cache instead of downloading models at pod startup.
