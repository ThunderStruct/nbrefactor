# Changelog

## 0.1.6 (2025-02-22)

[Compare the full difference.](https://github.com/ThunderStruct/nbrefactor/compare/0.1.3...0.1.6)

### Added

- Added new CLI flag `-i/--init-files` to optionally generate `__init__.py` files in package directories
- Improved package imports by properly exposing modules in `__init__.py` files

### Fixed

- Fixed relative import issues by properly handling package hierarchy with `__init__.py` files

## 0.1.3 (2025-02-18)

[Compare the full difference.](https://github.com/ThunderStruct/nbrefactor/compare/0.1.2...0.1.3)

### Added

- Added new `$analyze-only` command to handle global imports without creating unnecessary files/folders
- Improved handling of module declarations within package contexts

### Fixed

- Fixed an issue with imports cell creating unwanted `.py` files (primarily for root-level modules)
- Fixed module declaration within package contexts to properly create modules inside their parent packages

## 0.1.2 (2024-09-15)

_Initial version_

<!-- ## 0.1.1 (2024-09-15)
[Compare the full difference.](https://github.com/ThunderStruct/nbrefactor/compare/0.1.1...0.1.0)


### Added

- Added feature #1. [<commit-id>](<commit-link>) (<optionally add contributor name here>)
- Added feature #2. [<commit-id>](<commit-link>)
- Added feature #2. [<commit-id>](<commit-link>)


### Changed

- Changed feature #1. [<commit-id>](<commit-link>) (<optionally add contributor name here>)
- Changed feature #2. [<commit-id>](<commit-link>)
- Changed feature #2. [<commit-id>](<commit-link>)


### Fixed

- Fixed bug #1. [<commit-id>](<commit-link>) (<optionally add contributor name here>)
- Fixed bug #2. [<commit-id>](<commit-link>)
- Fixed bug #2. [<commit-id>](<commit-link>) -->

