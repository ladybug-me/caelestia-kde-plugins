# Caelestia Community Plugin Store

A community plugin store for [Caelestia](https://github.com/ladybug-me/caelestia-kde), the KDE Plasma 6 desktop shell.

Each folder under [`plugins/`](plugins/) is one plugin, contributed via pull request. CI checks the structure and a maintainer reviews every change before it merges.

Plugins are third-party code running with the same privileges as the desktop shell. Only source is accepted here - CI rejects compiled binaries and archives - and every change is reviewed by a human. Even so, only install plugins you trust. See [SECURITY.md](SECURITY.md) for more.

## Repository layout

| Path | Purpose |
| --- | --- |
| `plugins/<id>/` | One directory per plugin. The directory name is the plugin's unique, immutable id. |
| `plugins/<id>/metadata.json` | Required plugin manifest. |
| `plugins/<id>/LICENSE` | Required license file for the plugin. |
| `schemas/plugin.schema.json` | JSON Schema for `metadata.json`. |
| `scripts/validate.py` | Validates the whole store (run in CI and locally). |
| `scripts/build_index.py` | Regenerates `index.json`. |
| `index.json` | Generated registry consumed by the shell / other programs. |
| `docs/ingestion-contract.md` | Spec for programs that ingest this store. |
| `template-plugin/` | Copy-me scaffold for new plugins (not part of the store). |

## Adding a plugin

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version:

1. Fork this repository.
2. Copy `template-plugin/` to `plugins/<your-plugin-id>/` and fill it in.
3. Open a pull request. CI validates the structure; a maintainer reviews and merges.

## Consumers

If you're writing a program that reads this store, follow [`docs/ingestion-contract.md`](docs/ingestion-contract.md). Either clone the repo and read each `metadata.json`, or fetch `index.json` in a single request.

## Maintainers

- [@ladybug-me](https://github.com/ladybug-me)
- [@0xSolanaceae](https://github.com/0xSolanaceae)

## License

The repository's infrastructure (docs, scripts, schemas, workflows) is GPL-3.0 licensed - see [`LICENSE`](LICENSE). Each plugin folder is licensed independently, under whatever its own `LICENSE` file and `metadata.json` declare.
